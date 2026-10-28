"""3B.5B: real CPU metadata, v2 diagnostic precision and frozen v1 readers."""
from dataclasses import FrozenInstanceError
import inspect
import json
from pathlib import Path
import subprocess

import pytest
import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

from hyso import (audit_model_architecture, capture_model_structure, ModelArchitectureAudit,
                  ModelArchitectureAuditError, ModelStructureSnapshot)
from hyso.architecture_audit import ArchitectureCoverageStatus as Status
from hyso.analysis._codec import digest
from hyso.checkpoint import _rng_state
from test_architecture_audit import cnn, audit_only, at, issues
from test_model_structure import metadata_only, no_gpu_training_or_benchmark
from trainer_replay_support import equal

pytestmark = pytest.mark.usefixtures('no_gpu_training_or_benchmark')


def capture(monkeypatch, model):
    with metadata_only(monkeypatch): return capture_model_structure(model)


def run(monkeypatch, model):
    source = capture(monkeypatch, model)
    before = source.to_json()
    with audit_only(monkeypatch): result = audit_model_architecture(source)
    assert source.to_json() == before
    return result


def test_default_cnn_four_warnings_become_explicit_unassessed_areas(monkeypatch, tmp_path):
    result = run(monkeypatch, cnn())
    assert result.schema_version == 2 and result.audit_policy.version == 2
    assert result.audit_policy.id == 'registered_static_architecture_v2'
    assert inspect.signature(audit_model_architecture).parameters['policy_version'].default == 2
    assert result.summary['finding_counts'] == dict(STATIC_INCONSISTENCY=0, OBSERVATION=3, COVERAGE_GAP=0)
    assert result.evidence_gaps == ()
    assert len(result.findings) == 3
    assert {a.path for a in result.unassessed_areas} == {'', 'activation', 'pool', 'flatten'}
    assert all(a.status == Status.UNASSESSED_BY_RULESET for a in result.unassessed_areas)
    assert len(result.rule_applicability) == 12
    assert sum(a.status == Status.APPLICABLE for a in result.rule_applicability) == 2
    assert sum(a.status == Status.NOT_APPLICABLE for a in result.rule_applicability) == 10
    cov = result.coverage
    assert (cov['total_unique_modules'], cov['examined_supported_modules'], cov['unassessed_by_ruleset_modules']) == (6, 2, 4)
    assert cov['unassessed_module_ids'] == ['m0', 'm2', 'm3', 'm4']
    assert cov['status'] == 'scoped_static_evidence_assessed'
    assert cov['evidence_gap_count'] == 0 and cov['unassessed_area_count'] == 4
    assert all(a.status != 'consistent' for a in result.module_assessments if a.rule_id is None)
    assert {a.status for a in result.module_assessments} == {'consistent', 'unassessed_by_ruleset'}
    path = tmp_path / 'audit-v2.json'; result.to_json(path)
    result.source_snapshot.to_json(tmp_path / 'snapshot.json')
    with audit_only(monkeypatch):
        reopened = ModelArchitectureAudit.from_json(path)
        assert reopened.to_json() == result.to_json()
        assert audit_model_architecture(result.source_snapshot).to_json() == result.to_json()
    for row in (*result.rule_applicability, *result.unassessed_areas):
        assert row.source_snapshot_sha256 == result.source_snapshot.content_sha256
        assert all(f.value == at(result.source_snapshot.to_dict(), f.field) for f in row.observations)
    assert not {'winner', 'PASS', 'FAIL', 'certified_correct'} & result.to_dict().keys()
    (tmp_path / 'example-summary.json').write_text(json.dumps(dict(
        content_sha256=result.content_sha256, source_snapshot_sha256=result.source_snapshot.content_sha256,
        summary=result.summary, coverage=result.coverage), sort_keys=True), encoding='utf-8')


@pytest.mark.parametrize('kind', ['linear_weight', 'linear_bias', 'conv_weight', 'conv_bias', 'conv_channels',
                                  'conv_groups', 'conv_divisibility'])
def test_real_contradictions_and_observations_are_identical_to_v1(monkeypatch, kind):
    model = nn.Linear(3, 2) if kind.startswith('linear') else nn.Conv2d(4, 8, 3, groups=2)
    if kind == 'linear_weight': model.in_features = 9
    elif kind == 'linear_bias': model.bias = nn.Parameter(torch.empty(3))
    elif kind == 'conv_weight': model.weight = nn.Parameter(torch.empty(8, 4, 3, 3))
    elif kind == 'conv_bias': model.bias = nn.Parameter(torch.empty(9))
    elif kind == 'conv_channels': model.in_channels = 6
    elif kind == 'conv_groups': model.groups = 0
    else: model.groups = 3
    source = capture(monkeypatch, model)
    with audit_only(monkeypatch):
        old = audit_model_architecture(source, policy_version=1)
        new = audit_model_architecture(source)
    assert issues(new)
    for kind in ('STATIC_INCONSISTENCY', 'OBSERVATION'):
        assert [f for f in new.to_dict()['findings'] if f['classification'] == kind] == [
            f for f in old.to_dict()['findings'] if f['classification'] == kind]
    assert new.module_assessments == old.module_assessments
    assert new.evidence_gaps == ()


@pytest.mark.parametrize('kind', ['none_weight', 'missing_config', 'config_type', 'uninitialized_weight',
                                  'sparse_weight', 'meta_storage', 'mixed_known_and_missing'])
def test_only_required_missing_evidence_remains_a_coverage_gap(monkeypatch, kind):
    model = nn.Linear(3, 2, device='meta') if kind == 'meta_storage' else nn.Linear(3, 2)
    if kind == 'none_weight': model.weight = None
    elif kind == 'missing_config': del model.in_features
    elif kind == 'config_type': model.in_features = True
    elif kind == 'uninitialized_weight': model.weight = UninitializedParameter()
    elif kind == 'sparse_weight':
        model.weight = nn.Parameter(torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.]), (99, 99)))
    elif kind == 'mixed_known_and_missing': model.in_features = 9; model.bias = UninitializedParameter()
    result = run(monkeypatch, model)
    assert result.coverage['status'] == 'evidence_gaps_present' and result.evidence_gaps
    assert result.coverage['module_ids_with_evidence_gaps'] == ['m0']
    assert result.rule_applicability[0].status == Status.APPLICABLE
    assert result.unassessed_areas == ()
    assert bool(issues(result)) == (kind == 'mixed_known_and_missing')
    assert result.coverage['module_ids_with_insufficient_checks'] == ([] if kind == 'meta_storage' else ['m0'])
    findings = {f.finding_id: f for f in result.findings}
    for gap in result.evidence_gaps:
        assert gap.status == Status.COVERAGE_GAP
        finding = findings[gap.finding_id]
        assert finding.classification == 'COVERAGE_GAP' and finding.rule_version == 2
        assert gap.required_by_rule_id in ('linear_shapes', 'parameter_inventory', 'registration_relationships')
        assert gap.source_fields == tuple(f.field for f in finding.observations)


@pytest.mark.parametrize('kind', ['custom', 'subclass', 'spoofed_name', 'transposed', 'lazy'])
def test_unassessed_semantics_do_not_claim_missing_native_tensor_metadata(monkeypatch, kind):
    if kind == 'custom':
        class Custom(nn.Module):
            def __init__(self): super().__init__(); self.p = nn.Parameter(torch.ones(3))
        model = Custom()
    elif kind in ('subclass', 'spoofed_name'):
        class Custom(nn.Linear): pass
        if kind == 'spoofed_name': Custom.__module__, Custom.__qualname__ = 'torch.nn.modules.linear', 'Linear'
        model = Custom(3, 2); model.in_features = 100
    elif kind == 'transposed': model = nn.ConvTranspose2d(2, 4, 3)
    else: model = nn.LazyLinear(2)
    result = run(monkeypatch, model)
    assert not issues(result)
    assert result.module_assessments[0].status == 'unassessed_by_ruleset'
    assert all(r.status == Status.NOT_APPLICABLE for r in result.rule_applicability)
    assert len(result.unassessed_areas) == 1
    assert result.coverage['unassessed_by_ruleset_modules'] == 1
    assert bool(result.evidence_gaps) == (kind == 'lazy')
    if kind == 'lazy':
        assert {g.required_by_rule_id for g in result.evidence_gaps} == {'parameter_inventory', 'registration_relationships'}
        assert result.source_snapshot.summary.parameter_elements.value is None


@pytest.mark.parametrize('kind', ['conv_padding', 'pool_output'])
def test_unused_configuration_is_recorded_as_unassessed_not_required_evidence(monkeypatch, kind):
    model = nn.Conv2d(2, 4, 1) if kind == 'conv_padding' else nn.AdaptiveAvgPool2d(1)
    setattr(model, 'padding_mode' if kind == 'conv_padding' else 'output_size', object())
    result = run(monkeypatch, model)
    assert not issues(result) and result.evidence_gaps == ()
    areas = [a for a in result.unassessed_areas if a.area.startswith('configuration.')]
    assert len(areas) == 1 and areas[0].status == Status.UNASSESSED_BY_RULESET
    assert areas[0].observations[0].status == 'unsupported'
    assert areas[0].observations[0].value['status'] == 'unsupported'
    assert areas[0].reason == 'field_not_used_by_implemented_checks'
    if kind == 'conv_padding': assert result.module_assessments[0].status == 'consistent'


def test_tied_shared_storage_cycle_and_unique_counts_preserved(monkeypatch):
    model = nn.Module(); model.a = nn.Linear(3, 2); model.b = nn.Linear(3, 2)
    model.b.weight = model.a.weight; model.b.bias = model.a.bias
    model.reused = model.a; model.loop = model
    backing = torch.arange(8.); model.c = nn.Parameter(backing[:4]); model.d = nn.Parameter(backing[2:6])
    source = capture(monkeypatch, model)
    with audit_only(monkeypatch): old = audit_model_architecture(source, policy_version=1); new = audit_model_architecture(source)
    assert new.source_snapshot.summary.parameter_elements.value == 16
    assert [f for f in old.findings if f.rule_id == 'registration_relationships'] == [
        f for f in new.findings if f.rule_id == 'registration_relationships']
    assert new.evidence_gaps == () and new.coverage['total_unique_modules'] == 3


def test_frozen_v1_bytes_and_previous_reader_behavior(monkeypatch, tmp_path):
    source = capture(monkeypatch, cnn())
    previous = subprocess.run(['git', 'show',
        '7913c8e54c437469134be5285d970c186491998f:hyso/architecture_audit.py'],
        capture_output=True, text=True, check=True).stdout
    namespace = dict(__name__='hyso._previous_architecture_audit', __package__='hyso')
    exec(compile(previous, '<trusted-local-3b5a-audit>', 'exec'), namespace)
    with audit_only(monkeypatch):
        v1 = audit_model_architecture(source, policy_version=1)
        v2 = audit_model_architecture(source)
        old = namespace['audit_model_architecture'](source)
        assert v1.to_json() == old.to_json() and v1.content_sha256 == old.content_sha256
        assert namespace['ModelArchitectureAudit'](v1.to_dict()).to_json() == v1.to_json()
        assert ModelArchitectureAudit(v1.to_dict()).to_json() == v1.to_json()
        with pytest.raises(ValueError): namespace['ModelArchitectureAudit'](v2.to_dict())
        for name in ('rule_applicability', 'unassessed_areas', 'evidence_gaps'):
            with pytest.raises(ModelArchitectureAuditError, match='v2_coverage_required'): getattr(v1, name)
    assert v1.summary['finding_counts']['COVERAGE_GAP'] == 4
    assert v2.summary['finding_counts']['COVERAGE_GAP'] == 0
    assert v1.content_sha256 != v2.content_sha256
    v1.to_json(tmp_path / 'audit-v1.json'); v2.to_json(tmp_path / 'audit-v2.json')


def test_retained_3b5a_json_is_not_migrated_or_rehashed(monkeypatch):
    root = Path(__file__).resolve().parents[1] / 'outputs/architecture-audit-3b5a-20261028/example'
    audit_path, snapshot_path = root / 'audit.json', root / 'snapshot.json'
    if not audit_path.exists(): pytest.skip('retained local v1 acceptance artifact unavailable')
    before = audit_path.read_bytes()
    with audit_only(monkeypatch):
        reopened = ModelArchitectureAudit.from_json(audit_path)
        source = ModelStructureSnapshot.from_json(snapshot_path)
        assert audit_model_architecture(source, policy_version=1).to_json().encode('utf-8') == before
        assert reopened.to_json().encode('utf-8') == before
    assert audit_path.read_bytes() == before


@pytest.mark.parametrize('damage', ['applicability', 'unassessed_to_gap', 'drop_unassessed', 'scope_status',
    'source_value', 'source_identity', 'coverage_count', 'invent_warning', 'rule_version', 'policy_version',
    'downgrade', 'unsupported_schema', 'gap_reference', 'gap_required_rule', 'gap_module', 'gap_source_field'])
def test_reader_rederives_v2_dispositions_not_just_hashes(monkeypatch, damage):
    model = cnn()
    if damage.startswith('gap_'): model.head.weight = UninitializedParameter()
    d = run(monkeypatch, model).to_dict(); cov = d['coverage']
    if damage == 'applicability': cov['rule_applicability'][0]['status'] = 'APPLICABLE'
    elif damage == 'unassessed_to_gap': cov['unassessed_areas'][0]['status'] = 'COVERAGE_GAP'
    elif damage == 'drop_unassessed': cov['unassessed_areas'] = []
    elif damage == 'scope_status': cov['status'] = 'all_model_semantics_verified'
    elif damage == 'source_value': cov['unassessed_areas'][0]['observations'][0]['value'] = 'fabricated'
    elif damage == 'source_identity': cov['rule_applicability'][0]['source_snapshot_sha256'] = 'a' * 64
    elif damage == 'coverage_count': cov['unassessed_by_ruleset_modules'] = 0
    elif damage == 'invent_warning': d['findings'].append(dict(d['findings'][0], code='STATIC_SHAPE_RULE_NOT_IMPLEMENTED'))
    elif damage == 'rule_version': d['audit_policy']['rules'][-1]['version'] = 1
    elif damage == 'policy_version': d['audit_policy']['version'] = 1
    elif damage == 'downgrade': d['schema_version'] = 1
    elif damage == 'unsupported_schema': d['schema_version'] = True
    elif damage == 'gap_reference': cov['evidence_gaps'][0]['finding_id'] = 'a' * 64
    elif damage == 'gap_required_rule': cov['evidence_gaps'][0]['required_by_rule_id'] = 'convolution_shapes'
    elif damage == 'gap_module': cov['evidence_gaps'][0]['module_ids'] = ['m0']
    elif damage == 'gap_source_field': cov['evidence_gaps'][0]['source_fields'] = ['unobserved.native']
    d['content_sha256'] = digest({k: v for k, v in d.items() if k != 'content_sha256'})
    with audit_only(monkeypatch), pytest.raises(ModelArchitectureAuditError): ModelArchitectureAudit(d)


@pytest.mark.parametrize('version', [True, 0, 3, '2', None])
def test_closed_policy_versions(monkeypatch, version):
    source = capture(monkeypatch, cnn())
    with audit_only(monkeypatch), pytest.raises(ModelArchitectureAuditError, match='unsupported_architecture_audit_policy'):
        audit_model_architecture(source, policy_version=version)


def test_v2_typed_coverage_is_immutable_and_json_values_are_detached(monkeypatch):
    result = run(monkeypatch, cnn()); raw = result.to_json()
    with pytest.raises(FrozenInstanceError): result.rule_applicability[0].status = Status.APPLICABLE
    with pytest.raises(FrozenInstanceError): result.unassessed_areas[0].reason = 'invented'
    result.coverage['unassessed_areas'].clear()
    value = result.unassessed_areas[0].observations[0].value
    assert isinstance(value, str)
    assert result.to_json() == raw


def test_v2_preserves_original_model_state_code_modes_hooks_and_rng(monkeypatch):
    model = cnn(); model.train(); model.head.eval()
    model.register_buffer('owned_buffer', torch.ones(3))
    for p in model.parameters(): p.grad = torch.ones_like(p)
    called = []
    handles = [model.register_forward_hook(lambda *a: called.append('forward')),
               model.register_state_dict_pre_hook(lambda *a: called.append('state'))]
    def state():
        return ([t.detach().clone() for t in [*model.parameters(), *model.buffers()]],
            [(p._version, p.requires_grad, p.grad.clone()) for p in model.parameters()],
            [(m.training, tuple(m._forward_hooks), tuple(m._state_dict_pre_hooks)) for m in model.modules()],
            _rng_state(), nn.Sequential.forward.__code__.co_code)
    before = state(); source = capture(monkeypatch, model)
    with audit_only(monkeypatch):
        result = audit_model_architecture(source)
        assert ModelArchitectureAudit(result.to_dict()).to_json() == result.to_json()
    equal(before, state()); assert called == []
    for handle in handles: handle.remove()


def test_v2_extra_coverage_records_are_included_in_entry_budget(monkeypatch):
    source = capture(monkeypatch, cnn())
    with audit_only(monkeypatch):
        audit_model_architecture(source, policy_version=1, max_entries=100)
        with pytest.raises(ModelArchitectureAuditError, match='limit'):
            audit_model_architecture(source, max_entries=100)


@pytest.mark.parametrize('kind', ['invalid_groups_absent_weight', 'invalid_linear_absent_weight',
                                  'invalid_groups_lazy_weight', 'invalid_groups_lazy_bias'])
def test_invalid_prerequisite_does_not_require_undefined_shape_evidence(monkeypatch, kind):
    model = nn.Linear(3, 2) if kind == 'invalid_linear_absent_weight' else nn.Conv2d(4, 8, 3)
    if kind == 'invalid_linear_absent_weight': model.in_features = -1; model.weight = None
    else:
        model.groups = 0
        if kind == 'invalid_groups_absent_weight': model.weight = None
        elif kind == 'invalid_groups_lazy_weight': model.weight = UninitializedParameter()
        else: model.bias = UninitializedParameter()
    result = run(monkeypatch, model)
    assert issues(result)
    assert next(c for c in result.module_assessments[0].checks if c.code == 'weight_shape').status == 'not_applicable'
    assert not any(g.code == 'REQUIRED_PARAMETER_METADATA_MISSING' for g in result.evidence_gaps)
    assert bool(result.evidence_gaps) == ('lazy' in kind)
    if 'lazy' in kind:
        assert {g.required_by_rule_id for g in result.evidence_gaps} == {'parameter_inventory', 'registration_relationships'}
    if kind == 'invalid_groups_lazy_bias':
        assert next(c for c in result.module_assessments[0].checks if c.code == 'bias_shape').status == 'insufficient_evidence'
    else: assert result.coverage['module_ids_with_insufficient_checks'] == []
