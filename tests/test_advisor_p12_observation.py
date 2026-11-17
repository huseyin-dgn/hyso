"""New P12 product regressions on exact saved T05 bytes; zero live workloads.

Partial/missing cases are controlled in-memory reader mutations, not captures.
Each assertion describes only retained evidence or adapter behavior, not model
performance, hardware work, comparison eligibility or a new acceptance run.
"""
import copy
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path

import pytest
import torch
from hyso import advisor, analysis, performance
from hyso.analysis._codec import canonical, digest
from hyso.advisor.p12_observation import P12Observation, observe_p12
from hyso.advisor import AdvisorError, NativeMetricResolver, NativeMetric
from hyso.performance._operation_models import summarize

FIXTURES = Path(__file__).parent / 'fixtures'
SHA = {
    'p12-t05-sequential-v1.json': '358a1b14f72c00c7c15bca9dca7ace7a213a099e9d7aa2281c80bcc0553fc4bd',
    'p12-t05-graph-v1.json': '4cd9d108c091b07aff86fbb474f84a5034367fb906d5a5fbfe6e6cac1574a68b',
    'p14-t05-sequential-v1.json': 'dd7d32c0c8557d15172f6ab3e9e69c48722fcd9c353a6a311018a54469e82dad',
    'p16-t05-sequential-v1.json': '74788c3fd7df860698416ee255b8f7777a7284859a992bf1bf03d68fc65e40c4',
}


@pytest.fixture(autouse=True)
def no_workload(monkeypatch):
    """Any attempted forward/capture/counter/checkpoint/device work fails."""
    assert not torch.cuda.is_initialized()
    def forbidden(*a, **kw):
        pytest.fail('P12 observation attempted a forbidden live workload')
    for target, name in ((torch.nn.Module, '_call_impl'), (torch.nn.Module, '__init__'),
                         (torch, 'load'), (torch, 'save')):
        monkeypatch.setattr(target, name, forbidden)
    for name in ('init', '_lazy_init', 'is_available', 'device_count', 'current_device'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    for name in ('profile_operators', 'measure_utilization', 'measure_throughput'):
        monkeypatch.setattr(performance, name, forbidden)
    from hyso.performance import profiling, utilization, _telemetry
    monkeypatch.setattr(profiling, '_execute_profile', forbidden)
    monkeypatch.setattr(utilization, 'measure_utilization', forbidden)
    monkeypatch.setattr(_telemetry, 'cpu_observe', forbidden)
    yield
    assert not torch.cuda.is_initialized()


def saved(kind='sequential', feature='P12', scope=None):
    name = f'{feature.lower()}-t05-{kind}-v1.json'
    path = FIXTURES / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA[name]
    return analysis.EvidenceRecord.from_json(path, feature_id=feature,
        expected_sha256=SHA[name], scope=scope, origin='reused')


def report(records):
    return analysis.EvidenceIndex(records).analyze(context=analysis.AnalysisContext(
        workload_id='saved-P12-regression', task='retained evidence only', model={}, dataset={}))


@pytest.mark.parametrize('kind,flops,macs,calls', [
    ('sequential', 52490, 25856, 16), ('graph', 36106, 17664, 22)])
def test_saved_T05_observation_has_exact_derived_counts_and_call_coverage(kind, flops, macs, calls):
    """Batch1 CPU/FP32 observed counts only; no family/hardware acceptance."""
    r = saved(kind)
    d = observe_p12(r, version=1).to_dict()
    assert d['availability'] == 'AVAILABLE'
    assert d['metrics']['logical_flops_per_batch'] == dict(
        value=flops, unit='logical_FLOP/batch', status='DERIVED', reason=None)
    assert d['metrics']['macs_per_batch'] == dict(value=macs, unit='MAC/batch', status='DERIVED', reason=None)
    assert d['operator_coverage']['modeled_calls'] == d['operator_coverage']['observed_calls'] == calls
    assert d['operator_coverage']['unknown_calls'] == 0
    assert d['input']['shape'] == dict(value=[1, 1, 8, 8], authority='native')
    assert d['input']['batch'] == dict(value=1, authority='native')
    assert d['input']['dtype']['value'] == 'torch.float32'
    assert d['input']['device']['value'] == 'cpu'
    assert d['counting_convention']['id'] == 'aten_logical_v3'


@pytest.mark.parametrize('kind', ['sequential', 'graph'])
def test_native_identity_protocol_and_scope_are_retained_without_QA_substitution(kind):
    r = saved(kind); original = r.to_dict(); n = original['native_payload']
    d = observe_p12(r, version=1).to_dict()
    for name, value in (
        ('producer_result_id', n['result_id']),
        ('source_result_id', n['identity']['source_result_id']),
        ('source_payload_sha256', n['raw_evidence']['source_sha256']),
        ('trace_sha256', n['raw_evidence']['trace']['sha256'])):
        assert d['producer_identity'][name] == dict(value=value, authority='native')
    assert d['native_protocol'] == n['protocol']
    assert d['producer_identity']['native_protocol_sha256']['value'] == original['scope']['protocol_sha256']
    assert d['model_state']['value'] == n['provenance']['model']['weights_sha256']
    assert d['scope'] == original['scope'] and d['scope_bindings'] == original['bindings']
    assert d['scope_bindings']['model_id'] == 'missing'
    assert r.to_dict() == original


@pytest.mark.parametrize('feature', ['P14', 'P16'])
def test_real_wrong_native_feature_is_rejected_instead_of_relabelled(feature):
    with pytest.raises(AdvisorError, match='native_p12_operation_count_required'):
        observe_p12(saved(feature=feature), version=1)


def test_raw_result_or_unavailable_record_is_not_silently_adapted():
    from hyso.performance import OperationCountResult
    with pytest.raises(AdvisorError, match='p12_evidence_record_required'):
        observe_p12(OperationCountResult(saved().to_dict()['native_payload']), version=1)
    with pytest.raises(AdvisorError, match='native_p12_operation_count_required'):
        observe_p12(analysis.EvidenceRecord.unavailable('P12', reason='not captured'), version=1)


def test_partial_coverage_keeps_totals_null_and_original_subtotals_raw():
    """Unknown in-memory operator is a test mutation, never new native evidence."""
    n = copy.deepcopy(saved().to_dict()['native_payload'])
    n['raw_evidence']['aten_calls'][0]['operator'] = 'test_unmodeled.operator'
    rows, metrics = summarize(n['raw_evidence']['aten_calls'], 2, 1, model_version=3)
    n['raw_evidence']['operations'] = rows; n['measurements'] = metrics
    r = analysis.EvidenceRecord.from_native(n, feature_id='P12')
    d = observe_p12(r, version=1).to_dict()
    assert d['availability'] == 'PARTIAL' and d['operator_coverage']['unknown_calls'] == 1
    for m in d['metrics'].values():
        assert m['value'] is None and m['status'] == 'UNKNOWN'
        assert m['reason'] == 'unmodeled_observed_calls'
    assert d['evidence_record']['native_payload']['measurements']['supported_subtotal_flops'] > 0
    assert P12Observation.from_dict(d).to_dict() == d


def test_optional_producer_source_and_model_state_stay_missing():
    n = copy.deepcopy(saved().to_dict()['native_payload'])
    n['identity']['source_result_id'] = n['raw_evidence']['source_result_id'] = None
    n['provenance']['model'].pop('weights_sha256')
    r = analysis.EvidenceRecord.from_native(n, feature_id='P12')
    d = observe_p12(r, version=1).to_dict()
    assert d['producer_identity']['source_result_id'] == dict(value=None, authority='missing')
    assert d['model_state'] == dict(value=None, authority='missing')
    assert d['scope_bindings']['model_state_sha256'] == 'missing'
    assert d['producer_identity_status'] == 'PARTIAL'
    assert d['availability'] == 'AVAILABLE'  # Coverage and identity availability are different axes.


def test_mandatory_native_result_id_cannot_be_invented_by_adapter():
    n = copy.deepcopy(saved().to_dict()['native_payload']); n.pop('result_id')
    with pytest.raises(ValueError):analysis.EvidenceRecord.from_native(n, feature_id='P12')


def test_caller_only_QA_state_and_model_label_remain_declared():
    n = copy.deepcopy(saved().to_dict()['native_payload']); n['provenance']['model'].pop('weights_sha256')
    declared = analysis.EvidenceScope(model_id='qa-config:' + 'a' * 64, model_state_sha256='b' * 64)
    r = analysis.EvidenceRecord.from_native(n, feature_id='P12', scope=declared)
    d = observe_p12(r, version=1).to_dict()
    assert d['scope']['model_state_sha256'] == 'b' * 64
    assert d['scope_bindings']['model_state_sha256'] == d['scope_bindings']['model_id'] == 'declared'
    assert d['model_state'] == dict(value=None, authority='missing')
    assert d['declared_scope'] == declared.to_dict()
    with pytest.raises(ValueError, match='native_scope_mismatch:model_state_sha256'):
        saved(scope=analysis.EvidenceScope(model_state_sha256='b' * 64))


@pytest.mark.parametrize('version', [0, 2, True, 1.0, '1', None])
def test_unsupported_adapter_version_is_rejected(version):
    with pytest.raises(AdvisorError, match='unsupported_p12_observation_version'):
        observe_p12(saved(), version=version)


def test_version_is_required_and_json_version_does_not_fall_back():
    with pytest.raises(TypeError):observe_p12(saved())
    d = observe_p12(saved(), version=1).to_dict(); d['schema_version'] = 2
    with pytest.raises(AdvisorError, match='unsupported_p12_observation_schema'):P12Observation.from_dict(d)
    d['schema_version'] = 1; d['adapter_version'] = 2
    with pytest.raises(AdvisorError, match='unsupported_p12_observation_version'):P12Observation.from_dict(d)


@pytest.mark.parametrize('damage', ['counts', 'authority', 'claims', 'comparison', 'extra'])
def test_rehashed_projection_cannot_change_counts_authority_or_claims(damage):
    d = observe_p12(saved(), version=1).to_dict()
    if damage == 'counts':d['metrics']['logical_flops_per_batch']['value'] = 0
    elif damage == 'authority':d['scope_bindings']['model_id'] = 'native'
    elif damage == 'claims':d['claims']['global_winner'] = 'claimed'
    elif damage == 'comparison':d['comparison_eligibility'] = 'COMPARABLE'
    else:d['caller_extra'] = True
    d.pop('content_sha256'); d['content_sha256'] = digest(d)
    with pytest.raises(AdvisorError, match='derivation_or_integrity'):P12Observation.from_dict(d)


def test_JSON_roundtrip_immutable_detached_and_no_clobber(tmp_path):
    obj = observe_p12(saved(), version=1); data = obj.to_dict(); path = tmp_path / 'observation.json'
    assert obj.to_json(path) == path.read_text(encoding='utf8')
    assert P12Observation.from_json(path).to_dict() == data
    assert P12Observation.from_dict(data).to_json() == obj.to_json()
    with pytest.raises(FileExistsError):obj.to_json(path)
    with pytest.raises(FrozenInstanceError):obj._payload = 'forged'
    data['metrics'].clear(); assert obj.to_dict()['metrics']


def test_saved_artifact_bytes_unchanged_and_canonical_native_payload_exact(tmp_path):
    """Original file bytes stay exact; no promise to reproduce its whitespace."""
    for kind in ('sequential', 'graph'):
        path = FIXTURES / f'p12-t05-{kind}-v1.json'; before = path.read_bytes()
        r = saved(kind); payload = canonical(r.to_dict()['native_payload']).encode('utf8')
        obj = observe_p12(r, version=1); target = tmp_path / f'{kind}.json'; obj.to_json(target)
        restored = P12Observation.from_json(target).to_dict()
        assert canonical(restored['evidence_record']['native_payload']).encode('utf8') == payload
        assert restored['evidence_reference']['artifact']['sha256'] == hashlib.sha256(before).hexdigest()
        assert path.read_bytes() == before


def test_no_comparison_capture_or_performance_claim_is_fabricated():
    d = observe_p12(saved(), version=1).to_dict()
    assert 'comparison_capture' not in d['evidence_record']['native_payload']['provenance']
    assert d['comparison_eligibility'] == 'NOT_SUPPORTED_IN_COMPARISON'
    assert d['claims'] == dict(hardware_performance=False, model_comparison=False,
        model_performance_recommendation=False, global_winner=None, speedup=None,
        statistical_superiority='not_established', optimization_success='not_assessed')


def test_legacy_Advisor_resolution_and_JSON_are_identical_before_after_opt_in():
    payload = json.loads((FIXTURES / 'advisor-native-v1.json').read_text())['payloads']['P02']
    r = analysis.EvidenceRecord.from_native(payload, feature_id='P02'); rep = report([r])
    resolver = NativeMetricResolver(rep)
    before = resolver.resolve(r.record_id, NativeMetric.COMPLETED_IMAGES_PER_SECOND).to_dict()
    advice = advisor.advise(rep).to_json()
    observe_p12(saved(), version=1)
    assert resolver.resolve(r.record_id, NativeMetric.COMPLETED_IMAGES_PER_SECOND).to_dict() == before
    assert advisor.advise(rep).to_json() == advice and before['status'] == 'supported'
    p12 = saved(); legacy = NativeMetricResolver(report([p12]))
    assert legacy.resolve(p12.record_id, 'logical_flops_per_batch').status == 'unsupported_metric'
    from hyso.advisor._json import _TYPES
    assert 'P12Observation' not in _TYPES  # Existing schema decoder/registry is untouched.


@pytest.mark.parametrize('feature', ['P12', 'P16'])
def test_two_model_comparison_remains_unsupported_and_P16_raw_transport_unchanged(feature):
    rows = []
    for label in ('A', 'B'):
        r = saved(feature=feature, scope=analysis.EvidenceScope(model_id=label))
        rows.append(analysis.ModelComparisonSource(label, report([r]), records={feature:r.record_id}))
        if feature == 'P12':observe_p12(r, version=1)
        assert report([r]).to_dict()['index']['records'][0]['native_payload'] == r.to_dict()['native_payload']
    result = analysis.compare_models(*rows, features=feature).to_dict()
    assert result['metrics'][0]['status'] == 'NOT_SUPPORTED_IN_COMPARISON'
    assert result['metrics'][0]['difference'] is None


def test_existing_strict_and_legacy_comparison_schemas_and_results_stay_distinct():
    """Saved bound P26 payloads: no evaluation/model work is performed."""
    payloads = json.loads((FIXTURES / 'model-comparison-bound-v1.json').read_text())['payloads']
    sources = []
    for label in ('A', 'B'):
        scope = analysis.EvidenceScope(model_id=label, dataset_id=label, split='test', preprocessing_id='same')
        r = analysis.EvidenceRecord.from_native(payloads[label], feature_id='P26', scope=scope)
        sources.append(analysis.ModelComparisonSource(label, report([r]), records={'P26':r.record_id}))
    legacy = analysis.compare_models(*sources, features='P26')
    strict = analysis.compare_models(*sources, features='P26', context_policy='strict')
    observe_p12(saved(), version=1)
    assert analysis.compare_models(*sources, features='P26').to_json() == legacy.to_json()
    assert analysis.compare_models(*sources, features='P26', context_policy='strict').to_json() == strict.to_json()
    assert legacy.to_dict()['schema_version'] == 1 and strict.to_dict()['schema_version'] == 2
    assert all(v['status'] == 'COMPARABLE' for v in legacy.to_dict()['metrics'])
    assert all(v['status'] == 'INCOMPATIBLE' and v['difference'] is None for v in strict.to_dict()['metrics'])
