"""E.2: real schema-3 CPU producer -> resolver -> unified reader acceptance."""
from collections import OrderedDict
from contextlib import contextmanager
import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import subprocess

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyso import EvaluationResult, evaluate_classification, evaluate_predictions
from hyso.advisor import (AdvisorError, ComparisonContract, NativeMetric as M,
    NativeMetricResolver, P26QualityComparisonContract, P26BoundQualityComparisonContract,
    P26ModelProfileComparisonContract, UnifiedModelComparison, PerformanceComparison,
    compare_records, compare_performance, compare_p26_model_profiles)
from hyso.analysis import AnalysisContext, EvidenceIndex, EvidenceRecord, EvidenceScope
from hyso.analysis._codec import canonical, digest
from hyso.checkpoint import _rng_state, load_training_checkpoint, save_training_checkpoint
from hyso.structure_comparison import compare_model_structures
from test_model_structure import no_gpu_training_or_benchmark
from trainer_replay_support import equal


pytestmark = pytest.mark.usefixtures('no_gpu_training_or_benchmark')


def cnn(width=4, threshold=1.5, extra=False):
    parts = [('conv', nn.Conv2d(1, width, 1)), ('pool', nn.AdaptiveAvgPool2d(1)),
             ('flatten', nn.Flatten())]
    if extra: parts.append(('extra', nn.Identity()))
    parts.append(('head', nn.Linear(width, 2)))
    model = nn.Sequential(OrderedDict(parts))
    with torch.no_grad():
        model.conv.weight.fill_(1); model.conv.bias.zero_()
        model.head.weight[0].zero_(); model.head.weight[1].fill_(1 / width)
        model.head.bias.copy_(torch.tensor([0., -threshold]))
    model.conv.bias.requires_grad_(False)
    model.register_buffer('calibration', torch.tensor(1.))
    return model


def produce(instance=None, *, x=None, y=None, batch_size=2, **options):
    if x is None: x = torch.tensor([-2., -1., 1., 2.]).reshape(4, 1, 1, 1).expand(4, 1, 2, 2).clone()
    if y is None: y = torch.tensor([0, 0, 1, 1])
    config = dict(class_names=['negative', 'positive'], split='test',
                  capture_evidence=True, capture_structure=True)
    config.update(options)
    if config.get('task') == 'binary': y = y.float()
    return evaluate_classification(cnn() if instance is None else instance,
        DataLoader(TensorDataset(x, y), batch_size=batch_size), **config)


def pair(tmp_path, left=None, right=None, *, scopes=(None, None), separate=False):
    values = [produce() if left is None else left,
              produce(cnn(2, 0., True)) if right is None else right]
    records = []
    for i, (value, scope) in enumerate(zip(values, scopes)):
        path = tmp_path / ('source-' + str(i) + '.json')
        path.write_text(canonical(value.to_dict()), encoding='utf-8')
        records.append(EvidenceRecord.from_json(path, feature_id='P26', scope=scope,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    context = AnalysisContext(workload_id='bound-cnn', task='classification', model={}, dataset={})
    def resolver(items): return NativeMetricResolver(EvidenceIndex(items).analyze(context=context))
    resolvers = tuple(resolver([r]) for r in records) if separate else (resolver(records),) * 2
    return resolvers, records, values


def profile(resolvers, records, contract=None):
    return compare_p26_model_profiles(resolvers[0], records[0].record_id,
        resolvers[1], records[1].record_id,
        contract=P26ModelProfileComparisonContract(allowed_changes=('model',)) if contract is None else contract)


@contextmanager
def read_only(monkeypatch):
    import hyso.structure as structure
    import hyso._structure_native as native
    def forbidden(*args, **kwargs): raise AssertionError('comparison executed live model/capture/checkpoint')
    with monkeypatch.context() as patch:
        for name in ('_call_impl', 'train', 'eval', 'state_dict', 'load_state_dict',
                     'to', 'cpu', 'cuda', '_apply', 'register_forward_hook', 'register_forward_pre_hook'):
            patch.setattr(nn.Module, name, forbidden)
        patch.setattr(structure, 'capture_model_structure', forbidden)
        patch.setattr(native, 'capture', forbidden)
        patch.setattr(torch, 'load', forbidden)
        patch.setattr(torch.Tensor, 'backward', forbidden)
        for name in ('to', 'cpu', 'cuda'): patch.setattr(torch.Tensor, name, forbidden)
        yield


@pytest.mark.parametrize('separate', [False, True])
def test_real_cnn_producer_to_unified_json_reader(tmp_path, monkeypatch, separate):
    a, b = cnn(), cnn(2, 0., True)
    models = (a, b)
    before = [(copy.deepcopy(m.state_dict()), tuple(x.training for x in m.modules()),
               tuple(p.requires_grad for p in m.parameters())) for m in models]
    rng = _rng_state()
    left, right = produce(a), produce(b)
    rs, records, _ = pair(tmp_path, left, right, separate=separate)
    inputs = [r.to_dict() for r in records]
    reports = [r.report for r in rs]
    with read_only(monkeypatch):
        result = profile(rs, records)
        path = tmp_path / 'unified.json'; result.to_json(path)
        reopened = UnifiedModelComparison.from_json(path)
        assert reopened.to_json() == result.to_json()
        assert profile(rs, records).to_json() == result.to_json()
    for m, state in zip(models, before):
        equal(state, (m.state_dict(), tuple(x.training for x in m.modules()),
                      tuple(p.requires_grad for p in m.parameters())))
    equal(rng, _rng_state())
    assert inputs == [r.to_dict() for r in records] and reports == [r.report for r in rs]
    assert left.to_dict()['schema_version'] == right.to_dict()['schema_version'] == 3
    assert left.to_dict()['evaluation_id'] != right.to_dict()['evaluation_id']
    assert result.structure_eligibility == 'COMPARABLE'
    sc = result.structure_comparison
    assert sc.to_dict() == compare_model_structures(left.structure_snapshot, right.structure_snapshot).to_dict()
    summary = {d.metric: d for d in sc.summary_differences}
    for name, old, new in [('parameter_elements', 18, 10), ('trainable_elements', 14, 8),
                           ('frozen_elements', 4, 2), ('unique_modules', 5, 6)]:
        d = summary[name]
        assert (d.baseline.value, d.candidate.value, d.difference) == (old, new, new - old)
        assert d.percent_change == pytest.approx((new - old) / old * 100)
    assert sc.added_paths == ('extra',) and sc.removed_paths == ()
    assert {'conv', 'head'} <= {v.path for v in sc.changed_paths}
    conv = next(v for v in sc.changed_paths if v.path == 'conv')
    assert any(v.field == 'out_channels' for v in conv.configuration_changes)
    assert any(v.name == 'conv2d' for v in sc.category_differences)
    for i, (ref, evaluation) in enumerate(zip((result.baseline, result.candidate), (left, right))):
        d = evaluation.to_dict(); binding = d['structure_binding']
        assert ref.evaluation_id == d['evaluation_id']
        assert ref.record_id == records[i].record_id and ref.report_id == rs[i].source_report_id
        assert ref.structure_sha256 == evaluation.structure_snapshot.content_sha256
        assert ref.model_state_sha256 == d['producer_evidence']['model_state']['sha256']
        assert ref.model_state_sha256 != ref.structure_sha256
        assert ref.checkpoint_status == 'missing' and ref.checkpoint_sha256 is None
        assert binding['pre_structure_sha256'] == binding['post_structure_sha256'] == ref.structure_sha256
    for comp, old in [(result.accuracy_comparison, .75), (result.macro_f1_comparison, 11 / 15)]:
        assert comp.eligibility.status == 'COMPARABLE'
        d = comp.difference
        assert d.baseline.value == pytest.approx(old) and d.candidate.value == 1.
        assert d.percentage_point_change == pytest.approx((1 - old) * 100)
        assert d.percent_change == pytest.approx((1 - old) / old * 100)
        assert d.percentage_point_change != d.percent_change
        for i, m in enumerate((d.baseline, d.candidate)):
            assert m.reference.record_id == records[i].record_id
            assert m.reference.report_id == rs[i].source_report_id
            assert m.reference.protocol_ref.endswith(':schema3')
            assert rs[i].check_metric(m).status == 'supported'
        assert PerformanceComparison.from_dict(comp.to_dict()) == comp
    wire = result.to_dict()
    assert len(wire['source_reports']) == (2 if separate else 1)
    assert 'source_snapshots' not in wire['structure']['comparison']
    assert result.content_sha256 == digest({k: v for k, v in wire.items() if k != 'content_sha256'})
    assert result.contract.policy == 'p26_bound_model_profiles_v1'
    assert not {'status', 'winner', 'PASS', 'FAIL'} & wire.keys()
    assert 'causality_significance_and_overall_superiority_not_established' in result.limitations
    (tmp_path / 'example-summary.json').write_text(canonical(dict(
        baseline=wire['baseline'], candidate=wire['candidate'],
        parameter_elements=next(v for v in wire['structure']['comparison']['summary_differences']
                                if v['metric'] == 'parameter_elements'),
        accuracy=wire['quality']['accuracy']['comparison']['difference'],
        macro_f1=wire['quality']['macro_f1']['comparison']['difference'],
        content_sha256=result.content_sha256)), encoding='utf-8')


@pytest.mark.parametrize('change,field', [
    ('input', 'inputs_sha256'), ('target', 'targets_sha256'), ('pair', 'paired_sha256'),
    ('order', 'inputs_sha256'), ('count', 'sample_count'), ('batch', 'batch_sizes'),
    ('classes', 'configuration'), ('zero_division', 'configuration'), ('split', 'configuration')])
def test_quality_mismatch_preserves_valid_structure(tmp_path, change, field):
    x = torch.tensor([-2., -1., 1., 2.]).reshape(4, 1, 1, 1).expand(4, 1, 2, 2).clone()
    y = torch.tensor([0, 0, 1, 1]); options = {}
    if change == 'input': x[0] -= 1
    elif change == 'target': y[0] = 1
    elif change == 'pair': y = y.roll(1)
    elif change == 'order': x, y = x.flip(0), y.flip(0)
    elif change == 'count': x, y = x[:3], y[:3]
    elif change == 'batch': options['batch_size'] = 3
    elif change == 'classes': options['class_names'] = ['positive', 'negative']
    elif change == 'zero_division': options['zero_division'] = 1
    elif change == 'split': options['split'] = 'validation'
    rs, records, values = pair(tmp_path, produce(), produce(cnn(2, 0., True), x=x, y=y, **options))
    result = profile(rs, records)
    assert result.structure_comparison.baseline.content_sha256 == values[0].structure_snapshot.content_sha256
    assert result.structure_eligibility == 'COMPARABLE'
    for comp in (result.accuracy_comparison, result.macro_f1_comparison):
        assert comp.eligibility.status == 'INCOMPATIBLE' and comp.difference is None
        assert any(r.code == 'condition_mismatch' and r.field == field for r in comp.eligibility.reasons)
    assert UnifiedModelComparison(result.to_dict()).to_json() == result.to_json()


def test_permission_and_missing_quality_are_separate_from_structure(tmp_path):
    rs, records, _ = pair(tmp_path)
    for contract, status, reason in [
        (P26ModelProfileComparisonContract(), 'INCOMPATIBLE', 'unpermitted_change'),
        (P26ModelProfileComparisonContract(('model',), 5), 'INSUFFICIENT_EVIDENCE', 'sample_requirement_not_met')]:
        result = profile(rs, records, contract)
        assert result.structure_eligibility == 'COMPARABLE'
        for comp in (result.accuracy_comparison, result.macro_f1_comparison):
            assert comp.eligibility.status == status and comp.difference is None
            assert any(r.code == reason for r in comp.eligibility.reasons)
            assert comp.missing_evidence and comp.limitations


def test_equal_structure_different_weights_and_equal_state_do_not_certify_behavior(tmp_path):
    rs, records, values = pair(tmp_path, produce(cnn(2, 1.5)), produce(cnn(2, 0.)))
    result = profile(rs, records)
    assert result.baseline.structure_sha256 == result.candidate.structure_sha256
    assert result.baseline.model_state_sha256 != result.candidate.model_state_sha256
    assert result.accuracy_comparison.eligibility.status == 'COMPARABLE'
    # Two real evaluations of identical weights remain distinct captures, but
    # behavior identity is not established solely by equal type/state hashes.
    other = tmp_path / 'same-state'; other.mkdir()
    rs, records, _ = pair(other, values[0], produce(cnn(2, 1.5)))
    result = profile(rs, records, P26ModelProfileComparisonContract())
    assert result.baseline.evaluation_id != result.candidate.evaluation_id
    assert result.baseline.model_state_sha256 == result.candidate.model_state_sha256
    assert result.accuracy_comparison.eligibility.status == 'INSUFFICIENT_EVIDENCE'
    assert result.accuracy_comparison.difference is None


@pytest.mark.parametrize('reimport', [False, True])
def test_reused_evaluation_rejected_even_with_distinct_declarations(tmp_path, reimport):
    value = produce()
    scopes = (EvidenceScope(workload_id='a'), EvidenceScope(workload_id='b')) if reimport else (None, None)
    rs, records, _ = pair(tmp_path, value, value, scopes=scopes)
    if reimport: assert records[0].record_id != records[1].record_id
    with pytest.raises(AdvisorError, match='distinct_evaluations'):
        profile(rs, records)
    comp = compare_performance(rs[0], records[0].record_id, rs[1], records[1].record_id,
        contract=P26BoundQualityComparisonContract(allowed_changes=('model',)))
    assert comp.eligibility.status == 'INCOMPATIBLE' and comp.difference is None


@pytest.mark.parametrize('schema', [1, 2])
def test_old_schema_requires_its_original_explicit_policy(tmp_path, schema):
    left = produce(capture_structure=False, capture_evidence=schema == 2)
    right = produce(cnn(2, 0.), capture_structure=False, capture_evidence=schema == 2)
    rs, records, values = pair(tmp_path, left, right)
    before = [canonical(v.to_dict()) for v in values]
    with pytest.raises(AdvisorError, match='schema3'):
        profile(rs, records)
    for contract in [ComparisonContract(metric=M.TEST_ACCURACY),
                     P26BoundQualityComparisonContract(allowed_changes=('model',))]:
        comp = compare_performance(rs[0], records[0].record_id, rs[1], records[1].record_id, contract=contract)
        assert comp.eligibility.status == 'INSUFFICIENT_EVIDENCE' and comp.difference is None
    old = compare_performance(rs[0], records[0].record_id, rs[1], records[1].record_id,
                             contract=P26QualityComparisonContract(allowed_changes=('model',)))
    assert old.eligibility.status == ('COMPARABLE' if schema == 2 else 'INSUFFICIENT_EVIDENCE')
    assert before == [canonical(v.to_dict()) for v in values]


def test_schema3_never_silently_extends_old_contracts(tmp_path):
    rs, records, _ = pair(tmp_path)
    for contract in [ComparisonContract(metric=M.TEST_ACCURACY),
                     P26QualityComparisonContract(allowed_changes=('model',))]:
        eligibility = compare_records(rs[0], records[0].record_id, rs[1], records[1].record_id, contract=contract)
        assert eligibility.status == 'INSUFFICIENT_EVIDENCE'
    a = P26QualityComparisonContract(allowed_changes=('model',))
    b = P26BoundQualityComparisonContract(allowed_changes=('model',))
    assert a.content_sha256 != b.content_sha256 and a.to_dict()['format'] != b.to_dict()['format']
    assert type(b).from_dict(b.to_dict()) == b


@pytest.mark.parametrize('damage', ['missing', 'state', 'stream', 'run', 'checkpoint', 'snapshot'])
def test_corrupt_native_binding_rejected_after_rehash(damage):
    d = produce().to_dict(); b = d['structure_binding']
    if damage == 'missing': del d['structure_binding']
    elif damage == 'state': b['model_state_sha256'] = b['structure_sha256']
    elif damage == 'stream': b['evaluated_stream']['inputs_sha256'] = 'a' * 64
    elif damage == 'run': b['evaluation_id'] = 'unrelated-evaluation'
    elif damage == 'checkpoint': b['checkpoint'] = dict(status='native', sha256=b['model_state_sha256'])
    elif damage == 'snapshot': b['snapshot']['summary']['parameter_elements']['value'] += 1
    d['source_sha256'] = digest({k: v for k, v in d.items() if k not in ('source_sha256', 'metrics', 'status')})
    with pytest.raises(ValueError): EvaluationResult(d)


@pytest.mark.parametrize('damage', ['structure_value', 'quality_value', 'eligibility', 'reference',
    'baseline_state', 'candidate_evaluation', 'checkpoint', 'scope', 'limitations', 'source_key',
    'source_record', 'source_binding', 'policy', 'extra', 'version'])
def test_reader_rederives_sources_and_results_even_with_rehashed_envelope(tmp_path, damage):
    rs, records, _ = pair(tmp_path)
    d = profile(rs, records).to_dict()
    if damage == 'structure_value': d['structure']['comparison']['summary_differences'][0]['difference'] += 1
    elif damage == 'quality_value': d['quality']['accuracy']['comparison']['difference']['percentage_point_change'] += 1
    elif damage == 'eligibility': d['quality']['macro_f1']['comparison']['eligibility']['status'] = 'INCOMPATIBLE'
    elif damage == 'reference': d['quality']['accuracy']['comparison']['difference']['baseline']['reference']['record_id'] = records[1].record_id
    elif damage == 'baseline_state': d['baseline']['model_state_sha256'] = d['baseline']['structure_sha256']
    elif damage == 'candidate_evaluation': d['candidate']['evaluation_id'] = d['baseline']['evaluation_id']
    elif damage == 'checkpoint': d['candidate']['checkpoint_sha256'] = d['candidate']['model_state_sha256']
    elif damage == 'scope': d['quality']['accuracy']['scope'] = 'verified_generalization'
    elif damage == 'limitations': d['limitations'] = []
    elif damage == 'source_key': d['source_reports']['a' * 64] = d['source_reports'].pop(next(iter(d['source_reports'])))
    elif damage == 'source_record': d['baseline']['record_id'] = 'b' * 64
    elif damage == 'source_binding':
        report = next(iter(d['source_reports'].values()))
        report['index']['records'][0]['native_payload']['structure_binding']['model_state_sha256'] = 'a' * 64
    elif damage == 'policy': d['contract'] = P26ModelProfileComparisonContract().to_dict()
    elif damage == 'extra': d['winner'] = 'candidate'
    elif damage == 'version': d['schema_version'] = True
    # Deliberately also rehash nested comparisons: rejection must be from
    # source/result consistency, not just a stale checksum.
    def rehash(value):
        if isinstance(value, list):
            for item in value: rehash(item)
        elif isinstance(value, dict):
            for item in value.values(): rehash(item)
            if 'content_sha256' in value:
                value['content_sha256'] = digest({k: v for k, v in value.items() if k != 'content_sha256'})
    rehash(d)
    with pytest.raises(AdvisorError): UnifiedModelComparison(d)


def test_immutable_detached_and_exclusive_serialization(tmp_path):
    rs, records, _ = pair(tmp_path)
    result = profile(rs, records); raw = result.to_json()
    with pytest.raises(FrozenInstanceError): result._payload = '{}'
    with pytest.raises(FrozenInstanceError): result.baseline.evaluation_id = 'other'
    with pytest.raises(FrozenInstanceError): result.accuracy_comparison.warnings = ()
    with pytest.raises(FrozenInstanceError): result.structure_comparison._payload = '{}'
    data = result.to_dict(); data['source_reports'].clear()
    assert result.to_json() == raw
    path = tmp_path / 'result.json'; result.to_json(path)
    with pytest.raises(FileExistsError): result.to_json(path)
    assert path.read_text(encoding='utf-8') == raw
    assert UnifiedModelComparison.from_dict(result.to_dict()) == result
    path.write_text('{"format":1,"format":2}', encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate_json_key'): UnifiedModelComparison.from_json(path)


@pytest.mark.parametrize('contract_type', [P26BoundQualityComparisonContract, P26ModelProfileComparisonContract])
@pytest.mark.parametrize('kwargs', [dict(allowed_changes=('batch',)), dict(allowed_changes=('precision',)),
    dict(allowed_changes=('model', 'model')), dict(min_samples=True), dict(min_samples=0), dict(policy='other')])
def test_closed_versioned_contracts(contract_type, kwargs):
    with pytest.raises(AdvisorError): contract_type(**kwargs)


def test_checkpoint_artifact_is_retained_without_loading_during_comparison(tmp_path, monkeypatch):
    model = cnn()
    path = tmp_path / 'owned-checkpoint.pt'
    save_training_checkpoint(path, model)
    loaded = cnn(); load_training_checkpoint(path, loaded, capture_evidence=True)
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    rs, records, _ = pair(tmp_path, produce(loaded), produce(cnn(2, 0., True)))
    with read_only(monkeypatch):
        result = profile(rs, records)
        reopened = UnifiedModelComparison(result.to_dict())
    assert result.baseline.checkpoint_status == 'native'
    assert result.baseline.checkpoint_sha256 == expected_sha
    assert expected_sha not in (result.baseline.structure_sha256, result.baseline.model_state_sha256)
    assert reopened.candidate.checkpoint_sha256 is None
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha


def test_recorded_runtime_mismatch_and_declared_source_id_are_not_identity(tmp_path, monkeypatch):
    left = produce(source_id='same-caller-label')
    import hyso.evaluation as evaluation
    with monkeypatch.context() as patch:
        patch.setattr(evaluation.platform, 'python_version', lambda: '3.10.999')
        right = produce(cnn(2, 0.), source_id='same-caller-label')
    rs, records, _ = pair(tmp_path, left, right)
    result = profile(rs, records)
    assert result.structure_eligibility == 'COMPARABLE'
    assert result.baseline.evaluation_id != result.candidate.evaluation_id
    assert result.accuracy_comparison.eligibility.status == 'INCOMPATIBLE'
    assert result.accuracy_comparison.difference is None
    assert any(r.field == 'runtime' for r in result.accuracy_comparison.eligibility.reasons)


def test_binary_schema3_reuses_quality_semantics_and_zero_baseline(tmp_path):
    x = torch.tensor([[-2.], [-1.], [1.], [2.]])
    values = []
    for weight in (-1., 1.):
        model = nn.Linear(1, 1)
        with torch.no_grad(): model.weight.fill_(weight); model.bias.zero_()
        values.append(produce(model, x=x, task='binary'))
    rs, records, _ = pair(tmp_path, *values)
    result = profile(rs, records)
    for comp in (result.accuracy_comparison, result.macro_f1_comparison):
        assert comp.eligibility.status == 'COMPARABLE'
        assert comp.difference.baseline.value == 0. and comp.difference.candidate.value == 1.
        assert comp.difference.percentage_point_change == 100.
        assert comp.difference.percent_change is None
        assert 'zero_baseline_percent_undefined' in comp.warnings


def test_previous_e1_contract_and_schema2_result_canonical_bytes_unchanged(tmp_path, monkeypatch):
    from hyso.advisor import _json
    rs, records, _ = pair(tmp_path, produce(capture_structure=False),
                          produce(cnn(2, 0.), capture_structure=False))
    contract = P26QualityComparisonContract(allowed_changes=('model',))
    current = compare_performance(rs[0], records[0].record_id, rs[1], records[1].record_id, contract=contract)
    previous_source = subprocess.run(['git', 'show',
        'e1cd3490482f193fde6e9da3b31e01e940ff629c:hyso/advisor/performance.py'],
        capture_output=True, text=True, check=True).stdout
    monkeypatch.setattr(_json, '_TYPES', dict(_json._TYPES))
    namespace = {'__name__': 'hyso.advisor._e1_performance', '__package__': 'hyso.advisor'}
    exec(compile(previous_source, '<trusted-local-e1-performance>', 'exec'), namespace)
    old = namespace['compare_performance'](rs[0], records[0].record_id,
        rs[1], records[1].record_id, contract=contract)
    assert old.to_json() == current.to_json()
    assert old.content_sha256 == current.content_sha256
    previous_contracts = subprocess.run(['git', 'show',
        'e1cd3490482f193fde6e9da3b31e01e940ff629c:hyso/advisor/comparison.py'],
        capture_output=True, text=True, check=True).stdout
    namespace = {'__name__': 'hyso.advisor._e1_comparison', '__package__': 'hyso.advisor'}
    exec(compile(previous_contracts, '<trusted-local-e1-comparison>', 'exec'), namespace)
    assert namespace['P26QualityComparisonContract'](allowed_changes=('model',)).to_json() == contract.to_json()
    assert namespace['ComparisonContract']().to_json() == ComparisonContract().to_json()


def test_unified_closed_inputs_and_per_metric_types(tmp_path):
    rs, records, _ = pair(tmp_path)
    for resolvers, ids, contract in [
        ((None, rs[1]), (r.record_id for r in records), None),
        (rs, ('missing', records[1].record_id), None),
        (rs, (r.record_id for r in records), P26QualityComparisonContract()),
    ]:
        left, right = ids
        with pytest.raises(AdvisorError):
            compare_p26_model_profiles(resolvers[0], left, resolvers[1], right, contract=contract)
    with pytest.raises(AdvisorError): P26BoundQualityComparisonContract(metric=M.FORWARD_P95)


def test_unobservable_quality_stays_missing_and_result_size_is_bounded(tmp_path, monkeypatch):
    rs, records, _ = pair(tmp_path, produce(split='validation'), produce(cnn(2, 0.), split='validation'))
    result = profile(rs, records)
    assert result.structure_eligibility == 'COMPARABLE'
    for comp in (result.accuracy_comparison, result.macro_f1_comparison):
        assert comp.eligibility.status == 'INSUFFICIENT_EVIDENCE' and comp.difference is None
        assert any(r.code == 'metric_not_observable' for r in comp.eligibility.reasons)
    import hyso.advisor.profiles as profiles
    monkeypatch.setattr(profiles, '_MAX_BYTES', 100)
    with pytest.raises(AdvisorError, match='size_limit'): UnifiedModelComparison(result.to_dict())
