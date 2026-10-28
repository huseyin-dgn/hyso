"""3B.4C.2: real CPU P26 producer -> native reader -> resolver -> comparison."""
import copy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hyso import EvaluationResult, evaluate_classification, evaluate_predictions
from hyso.analysis import AnalysisContext, EvidenceIndex, EvidenceRecord, EvidenceScope, EvidenceError
from hyso.advisor import (AdvisorError, ComparisonContract, NativeMetric as M, NativeMetricResolver,
                          P26QualityComparisonContract, PerformanceComparison,
                          compare_records, compare_performance)
from hyso.advisor._json import canonical
from hyso.checkpoint import load_training_checkpoint, save_training_checkpoint, _rng_state
from hyso.evaluation import _hash
from trainer_replay_support import equal


@pytest.fixture(autouse=True)
def no_training_benchmark_or_gpu(monkeypatch):
    from hyso.benchmark import BenchmarkRunner
    from hyso.training import Trainer
    def forbidden(*args, **kwargs):
        raise AssertionError('training/benchmark/GPU call forbidden')
    for name in ('init', 'is_available', 'device_count', 'current_device'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(Trainer, 'fit', forbidden)
    monkeypatch.setattr(BenchmarkRunner, 'run', forbidden)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def model(threshold=0., *, binary=False):
    result = nn.Linear(1, 1 if binary else 2)
    with torch.no_grad():
        result.weight.copy_(torch.tensor([[1.]] if binary else [[0.], [1.]]))
        result.bias.copy_(torch.tensor([-threshold] if binary else [0., -threshold]))
    return result


def produce(threshold=0., *, instance=None, x=None, y=None, batch_size=2, **options):
    config = dict(class_names=['negative', 'positive'], split='test', capture_evidence=True)
    config.update(options)
    binary = config.get('task') == 'binary'
    if x is None: x = torch.tensor([[-2.], [-1.], [1.], [2.]])
    if y is None: y = torch.tensor([0, 0, 1, 1])
    if binary: y = y.float()
    return evaluate_classification(instance if instance is not None else model(threshold, binary=binary),
        DataLoader(TensorDataset(x, y), batch_size=batch_size), **config)


def pair(tmp_path, left=None, right=None, *, scopes=(None, None), features=('P26', 'P26')):
    values = [produce(1.5) if left is None else left, produce() if right is None else right]
    records = []
    for i, (value, scope, feature) in enumerate(zip(values, scopes, features)):
        value = value.to_dict() if isinstance(value, EvaluationResult) else value
        raw = canonical(value).encode('utf-8')
        path = tmp_path / (str(i) + '.json'); path.write_bytes(raw)
        records.append(EvidenceRecord.from_json(path, feature_id=feature,
            expected_sha256=hashlib.sha256(raw).hexdigest(), scope=scope))
    report = EvidenceIndex(records).analyze(context=AnalysisContext(
        workload_id='p26-observed-fixture', task='classification', model={}, dataset={}))
    return NativeMetricResolver(report), records


def policy(metric=M.TEST_ACCURACY, **kwargs):
    options = dict(metric=metric, allowed_changes=('model',))
    options.update(kwargs)
    return P26QualityComparisonContract(**options)


def compare(resolver, records, contract=None):
    return compare_performance(resolver, records[0].record_id, resolver, records[1].record_id,
                               contract=policy() if contract is None else contract)


@pytest.mark.parametrize('metric', [M.TEST_ACCURACY, M.TEST_MACRO_F1])
@pytest.mark.parametrize('task', ['multiclass', 'binary'])
def test_real_producer_positive_quality_comparison(tmp_path, monkeypatch, metric, task):
    a, b = model(1.5, binary=task == 'binary'), model(binary=task == 'binary')
    before = (copy.deepcopy(a.state_dict()), copy.deepcopy(b.state_dict()), _rng_state())
    left, right = produce(instance=a, task=task), produce(instance=b, task=task)
    equal(before, (a.state_dict(), b.state_dict(), _rng_state()))
    resolver, records = pair(tmp_path, left, right)
    # Comparison consumes only recorded native data, never the model objects.
    def forbidden(*args, **kwargs): raise AssertionError('comparison executed forward')
    monkeypatch.setattr(nn.Module, '_call_impl', forbidden)
    result = compare(resolver, records, policy(metric))
    assert result.eligibility.status == 'COMPARABLE'
    difference = result.difference
    baseline = .75 if metric == M.TEST_ACCURACY else float(Fraction(11, 15))
    assert difference.baseline.value == pytest.approx(baseline)
    assert difference.candidate.value == 1.
    assert difference.difference == pytest.approx(1 - baseline)
    assert difference.percentage_point_change == pytest.approx(100 * (1 - baseline))
    assert difference.percent_change == pytest.approx(100 * (1 - baseline) / baseline)
    assert difference.metric_effect == 'favorable'
    assert result.contract.policy == 'p26_exact_observed_stream_v1'
    assert result.eligibility.contract_sha256 == result.contract.content_sha256
    assert difference.baseline.reference == result.eligibility.left
    assert difference.candidate.reference == result.eligibility.right
    for i, metric in enumerate((difference.baseline, difference.candidate)):
        assert metric.reference.record_id == records[i].record_id
        assert metric.reference.artifact_sha256 == records[i].to_dict()['artifact']['sha256']
        assert metric.reference.protocol_ref == 'hyso.evaluation.classification:schema2'
    assert 'single_metric_only' in result.limitations
    assert 'P29_independent_validation_incomplete' in result.limitations
    assert 'statistical_significance_not_assessed' in result.limitations
    assert result.eligibility.statistical_independence == 'not_certified'
    assert PerformanceComparison.from_dict(result.to_dict()) == result


@pytest.mark.parametrize('metric', [M.TEST_ACCURACY, M.TEST_MACRO_F1])
def test_signed_decrease_and_zero_baseline(tmp_path, metric):
    x = torch.tensor([[-2.], [-1.], [1.], [2.]])
    inverse = model()
    with torch.no_grad(): inverse.weight.neg_()
    r, records = pair(tmp_path, produce(instance=inverse), produce(x=x))
    difference = compare(r, records, policy(metric)).difference
    assert difference.baseline.value == 0.
    assert difference.difference == 1. and difference.percentage_point_change == 100.
    assert difference.percent_change is None
    assert 'zero_baseline_percent_undefined' in compare(r, records, policy(metric)).warnings
    reverse = compare_performance(r, records[1].record_id, r, records[0].record_id, contract=policy(metric)).difference
    assert reverse.difference == -1. and reverse.percentage_point_change == -100.
    assert reverse.percent_change == -100. and reverse.metric_effect == 'unfavorable'


@pytest.mark.parametrize('change,field', [
    ('input', 'inputs_sha256'), ('target', 'targets_sha256'), ('pairing', 'paired_sha256'),
    ('count', 'sample_count'), ('order', 'inputs_sha256'),
    ('classes', 'configuration'), ('zero_division', 'configuration'),
    ('task', 'configuration'), ('batch', 'batch_sizes')])
def test_actual_producer_stream_and_protocol_mismatches_rejected(tmp_path, change, field):
    kwargs = {}
    x = torch.tensor([[-2.], [-1.], [1.], [2.]]); y = torch.tensor([0, 0, 1, 1])
    if change == 'input': x[0, 0] -= 1
    elif change == 'target': y[0] = 1
    elif change == 'pairing': y = y.roll(1)
    elif change == 'count': x, y = x[:3], y[:3]
    elif change == 'order': x, y = x.flip(0), y.flip(0)
    elif change == 'classes': kwargs['class_names'] = ['positive', 'negative']
    elif change == 'zero_division': kwargs['zero_division'] = 1
    elif change == 'task': kwargs['task'] = 'binary'
    elif change == 'batch': kwargs['batch_size'] = 3
    r, records = pair(tmp_path, produce(1.5), produce(x=x, y=y, **kwargs))
    result = compare(r, records)
    assert result.eligibility.status == 'INCOMPATIBLE' and result.difference is None
    assert any(reason.field == field for reason in result.eligibility.reasons)
    if change == 'batch':
        # Content identity ignores batch boundaries; quality protocol v1 does not.
        streams = [record.to_dict()['native_payload']['producer_evidence']['evaluated_stream'] for record in records]
        assert streams[0] == streams[1]


def test_model_permission_is_explicit_and_named_tensors_do_not_prove_behavior(tmp_path):
    r, records = pair(tmp_path)
    result = compare(r, records, policy(allowed_changes=()))
    assert result.eligibility.status == 'INCOMPATIBLE'
    assert any(reason.code == 'unpermitted_change' for reason in result.eligibility.reasons)
    shared = model()
    r, records = pair(tmp_path, produce(instance=shared), produce(instance=shared))
    assert compare(r, records).eligibility.status == 'COMPARABLE'
    result = compare(r, records, policy(allowed_changes=()))
    assert result.eligibility.status == 'INSUFFICIENT_EVIDENCE'
    assert any(reason.code == 'model_behavior_identity_not_captured' for reason in result.eligibility.reasons)


@pytest.mark.parametrize('reuse', ['same_record', 'same_run', 'reimport'])
def test_reused_underlying_capture_rejected(tmp_path, monkeypatch, reuse):
    if reuse == 'same_run':
        monkeypatch.setattr(uuid, 'uuid4', lambda: uuid.UUID('20000000-0000-0000-0000-000000000000'))
    a, b = produce(1.5), produce()
    if reuse == 'reimport': b = a
    scopes = (None, EvidenceScope(dataset_id='different-import-label')) if reuse == 'reimport' else (None, None)
    r, records = pair(tmp_path, a, b, scopes=scopes)
    right_id = records[0].record_id if reuse == 'same_record' else records[1].record_id
    result = compare_performance(r, records[0].record_id, r, right_id, contract=policy())
    assert result.eligibility.status == 'INCOMPATIBLE' and result.difference is None
    assert any(reason.code == 'same_underlying_capture' for reason in result.eligibility.reasons)


def test_missing_checkpoint_origin_and_preprocessing_not_repaired_or_required(tmp_path):
    a, b = produce(1.5, source_id='caller-left'), produce(source_id='caller-right')
    scopes = (EvidenceScope(checkpoint_sha256='a'*64, dataset_id='declared-left', preprocessing_id='left'),
              EvidenceScope(checkpoint_sha256='b'*64, dataset_id='declared-right', preprocessing_id='right'))
    r, records = pair(tmp_path, a, b, scopes=scopes)
    before = canonical(r.report)
    result = compare(r, records)
    assert result.eligibility.status == 'COMPARABLE'
    for record in records:
        d = record.to_dict(); e = d['native_payload']['producer_evidence']
        for key in ('split', 'checkpoint_sha256', 'dataset_id', 'preprocessing_id'):
            assert d['bindings'][key] == 'declared'
        assert d['bindings']['model_state_sha256'] == 'native'
        assert e['checkpoint']['status'] == e['preprocessing']['status'] == e['dataset_origin']['status'] == 'missing'
        assert e['source_id']['status'] == e['split']['status'] == 'declared'
    for limit in ('dataset_origin_not_verified', 'preprocessing_implementation_not_verified',
                  'checkpoint_artifact_not_required_or_inferred_from_model_state', 'split_and_source_id_are_caller_declared'):
        assert limit in result.limitations
    assert canonical(r.report) == before


@pytest.mark.parametrize('captured', [False, True])
def test_optional_checkpoint_link_stays_separate_from_state(tmp_path, captured):
    a, b = model(1.5), model()
    path = tmp_path / 'trusted.pth'
    save_training_checkpoint(path, a)
    load_training_checkpoint(path, a, capture_evidence=captured)
    r, records = pair(tmp_path, produce(instance=a), produce(instance=b))
    assert compare(r, records).eligibility.status == 'COMPARABLE'
    d = records[0].to_dict()
    assert d['bindings']['checkpoint_sha256'] == ('native' if captured else 'missing')
    if captured:
        assert d['scope']['checkpoint_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert d['scope']['checkpoint_sha256'] != d['scope']['model_state_sha256']


def test_general_contract_still_requires_its_original_native_metadata(tmp_path):
    r, records = pair(tmp_path)
    for metric in (M.TEST_ACCURACY, M.TEST_MACRO_F1):
        result = compare(r, records, ComparisonContract(metric=metric, allowed_changes=('model',), min_samples=1))
        assert result.eligibility.status == 'INSUFFICIENT_EVIDENCE' and result.difference is None


def test_quality_policy_does_not_apply_to_latency_records(tmp_path):
    from test_advisor_comparison import payload
    r, records = pair(tmp_path, payload('1', 10.), payload('2', 7.), features=('P01', 'P01'))
    result = compare(r, records)
    assert result.eligibility.status == 'INSUFFICIENT_EVIDENCE' and result.difference is None


def test_equal_declared_source_ids_do_not_replace_native_run_identity(tmp_path):
    r, records = pair(tmp_path, produce(1.5, source_id='same-declared-cohort'), produce(source_id='same-declared-cohort'))
    assert compare(r, records).eligibility.status == 'COMPARABLE'
    assert records[0].to_dict()['native_payload']['evaluation_id'] != records[1].to_dict()['native_payload']['evaluation_id']


@pytest.mark.parametrize('kind', ['schema1', 'supplied', 'empty', 'not_test', 'unavailable', 'unknown_id', 'minimum'])
def test_insufficient_evidence_has_no_numeric_difference(tmp_path, kind):
    left = produce(1.5)
    if kind == 'schema1': right = produce(capture_evidence=False)
    elif kind == 'supplied': right = evaluate_predictions([0, 0, 1, 1], [0, 0, 1, 1], class_names=['negative', 'positive'], split='test')
    elif kind == 'empty': right = produce(x=torch.empty(0, 1), y=torch.empty(0, dtype=torch.long))
    elif kind == 'not_test': right = produce(split='validation')
    else: right = produce()
    r, records = pair(tmp_path, left, right)
    if kind == 'unavailable':
        records[1] = EvidenceRecord.unavailable('P26', reason='not captured')
        r = NativeMetricResolver(EvidenceIndex(records).analyze(context=AnalysisContext(workload_id='missing', task='classification', model={}, dataset={})))
    right_id = 'f'*64 if kind == 'unknown_id' else records[1].record_id
    result = compare_performance(r, records[0].record_id, r, right_id, contract=policy(min_samples=5 if kind == 'minimum' else 1))
    # Known count/configuration differences take precedence over missing fields.
    expected = 'INCOMPATIBLE' if kind in ('empty', 'not_test') else 'INSUFFICIENT_EVIDENCE'
    assert result.eligibility.status == expected and result.difference is None


def rehash(data):
    data['source_sha256'] = _hash({k: v for k, v in data.items() if k not in ('source_sha256', 'metrics', 'status')})
    return data


@pytest.mark.parametrize('field,value', [('method', 'repr'), ('scope', 'unordered_dataset'),
    ('sample_count', 7), ('status', 'declared'), ('batch_partition_invariant', False)])
def test_invalid_stream_evidence_rejected_at_native_reader(tmp_path, field, value):
    data = produce().to_dict()
    data['producer_evidence']['evaluated_stream'][field] = value
    with pytest.raises(EvidenceError, match='invalid_native_evidence'):
        pair(tmp_path, produce(1.5), rehash(data))


@pytest.mark.parametrize('field,value', [('status', 'declared'), ('sha256', None)])
def test_declared_or_missing_model_state_cannot_be_repaired_by_scope(tmp_path, field, value):
    data = produce().to_dict()
    state = data['producer_evidence']['model_state']['sha256']
    data['producer_evidence']['model_state'][field] = value
    with pytest.raises(EvidenceError, match='invalid_native_evidence'):
        pair(tmp_path, produce(1.5), rehash(data), scopes=(None, EvidenceScope(model_state_sha256=state)))


def test_isolated_pair_hash_mismatch_never_ignored(tmp_path):
    data = produce().to_dict()
    data['producer_evidence']['evaluated_stream']['paired_sha256'] = 'a'*64
    r, records = pair(tmp_path, produce(1.5), rehash(data))
    result = compare(r, records)
    assert result.eligibility.status == 'INCOMPATIBLE' and result.difference is None
    assert any(reason.field == 'paired_sha256' for reason in result.eligibility.reasons)


@pytest.mark.parametrize('field', ['binary_rule', 'multiclass_rule'])
def test_unsupported_prediction_rule_rejected_by_native_reader(tmp_path, field):
    data = produce().to_dict(); data['configuration'][field] = 'caller alternative'
    with pytest.raises(EvidenceError, match='invalid_native_evidence'):
        pair(tmp_path, produce(1.5), rehash(data))


def test_runtime_mismatch_and_unknown_model_type_fail_closed(tmp_path):
    data = produce().to_dict(); data['provenance']['torch'] = 'other-runtime'
    r, records = pair(tmp_path, produce(1.5), rehash(data))
    assert compare(r, records).eligibility.status == 'INCOMPATIBLE'
    data = produce().to_dict(); data['source']['model_type'] = 'unknown'
    r, records = pair(tmp_path, produce(1.5), rehash(data))
    assert compare(r, records).eligibility.status == 'INSUFFICIENT_EVIDENCE'


@pytest.mark.parametrize('kwargs', [dict(metric=M.FORWARD_P95), dict(metric=M.COMPLETED_IMAGES_PER_SECOND),
    dict(metric=M.RSS_CURRENT), dict(allowed_changes=('batch',)), dict(allowed_changes=('precision',)),
    dict(allowed_changes=('model', 'model')), dict(min_samples=0), dict(min_samples=True), dict(policy='v2')])
def test_policy_is_closed_and_cannot_relax_general_performance(kwargs):
    with pytest.raises(AdvisorError): P26QualityComparisonContract(**kwargs)


def test_contract_roundtrip_tampering_and_boundary_immutability(tmp_path):
    p = policy(); assert P26QualityComparisonContract.from_dict(p.to_dict()) == p
    with pytest.raises(FrozenInstanceError): p.policy = 'other'
    p.to_json(tmp_path / 'policy.json')
    assert P26QualityComparisonContract.from_json(tmp_path / 'policy.json') == p
    r, records = pair(tmp_path); result = compare(r, records)
    assert result.to_json() == compare(r, records).to_json()
    result.to_json(tmp_path / 'comparison.json')
    assert PerformanceComparison.from_json(tmp_path / 'comparison.json') == result
    with pytest.raises(AdvisorError): replace(result, limitations=())
    with pytest.raises(AdvisorError): replace(result, missing_evidence=())
    with pytest.raises(AdvisorError): replace(result, contract=policy(M.TEST_MACRO_F1))
    for field in ('schema_version', 'policy'):
        data = p.to_dict(); data[field] = 2 if field == 'schema_version' else 'other'
        with pytest.raises(AdvisorError): P26QualityComparisonContract.from_dict(data)


def test_previous_general_contract_canonical_bytes_are_unchanged(monkeypatch):
    from hyso.advisor import _json
    monkeypatch.setattr(_json, '_TYPES', dict(_json._TYPES))
    source = subprocess.run(['git', 'show', '7f353d8ce778aff4bcbf2c698d0fda09a6a24a03:hyso/advisor/comparison.py'],
                            check=True, capture_output=True, text=True).stdout
    namespace = {'__name__': 'hyso.advisor._previous_comparison', '__package__': 'hyso.advisor'}
    exec(compile(source, '<trusted-previous-comparison>', 'exec'), namespace)
    for metric in (M.FORWARD_P95, M.COMPLETED_IMAGES_PER_SECOND, M.RSS_CURRENT, M.TEST_ACCURACY):
        kwargs = dict(metric=metric, allowed_changes=('model',), min_samples=1)
        assert ComparisonContract(**kwargs).to_json() == namespace['ComparisonContract'](**kwargs).to_json()


def test_previous_performance_json_unchanged_and_old_reader_rejects_new_policy(tmp_path, monkeypatch):
    from test_advisor_comparison import payload
    from hyso.advisor import _json
    r, records = pair(tmp_path, payload('1', 10.), payload('2', 7.), features=('P01', 'P01'))
    general = ComparisonContract(min_samples=20, min_tail_samples=1)
    current = compare(r, records, general)
    pr, p_records = pair(tmp_path, produce(1.5), produce())
    new = compare(pr, p_records)
    old_source = subprocess.run(['git', 'show', '7f353d8ce778aff4bcbf2c698d0fda09a6a24a03:hyso/advisor/performance.py'],
                                check=True, capture_output=True, text=True).stdout
    monkeypatch.setattr(_json, '_TYPES', dict(_json._TYPES))
    namespace = {'__name__': 'hyso.advisor._previous_performance', '__package__': 'hyso.advisor'}
    exec(compile(old_source, '<trusted-previous-performance>', 'exec'), namespace)
    previous = namespace['compare_performance'](r, records[0].record_id, r, records[1].record_id, contract=general)
    assert current.to_json() == previous.to_json()
    assert namespace['PerformanceComparison'].from_dict(current.to_dict()).to_json() == current.to_json()
    with pytest.raises(AdvisorError): namespace['PerformanceComparison'].from_dict(new.to_dict())
