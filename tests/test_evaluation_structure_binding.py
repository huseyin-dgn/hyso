"""E.1: actual isolated P26 receiver -> snapshot -> schema3 -> reader/resolver."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

from hyso import evaluate_classification, EvaluationResult, ModelStructureSnapshot
from hyso.analysis import AnalysisContext, EvidenceIndex, EvidenceRecord, EvidenceScope, EvidenceError
from hyso.advisor import (NativeMetric as M, NativeMetricResolver, P26QualityComparisonContract,
                          ComparisonContract, compare_performance)
from hyso.checkpoint import save_training_checkpoint, load_training_checkpoint, _rng_state
from hyso.evaluation import _hash, _model_hash
from hyso.analysis._codec import canonical, digest
from test_model_structure import metadata_only, no_gpu_training_or_benchmark
from trainer_replay_support import equal


def loader():
    return DataLoader(TensorDataset(torch.arange(15, dtype=torch.float32).reshape(5, 3) / 20,
                                   torch.tensor([0, 1, 0, 1, 0])), batch_size=2)


def produce(model=None, data=None, **kwargs):
    options = dict(class_names=['a', 'b'], split='test', capture_evidence=True, capture_structure=True)
    options.update(kwargs)
    return evaluate_classification(nn.Linear(3, 2) if model is None else model,
                                   loader() if data is None else data, **options)


def record(tmp_path, result, name='evaluation', scope=None):
    path = tmp_path / (name + '.json'); result.to_json(path)
    return EvidenceRecord.from_json(path, feature_id='P26', scope=scope,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def resolver(records):
    report = EvidenceIndex(records).analyze(context=AnalysisContext(
        workload_id='bound-structure-fixture', task='classification', model={}, dataset={}))
    return NativeMetricResolver(report)


def reader_only(monkeypatch):
    import hyso.structure as structure
    import hyso._structure_native as native
    def forbidden(*args, **kwargs): raise AssertionError('reader executed model/capture/checkpoint')
    monkeypatch.setattr(nn.Module, '_call_impl', forbidden)
    monkeypatch.setattr(structure, 'capture_model_structure', forbidden)
    monkeypatch.setattr(native, 'capture', forbidden)
    monkeypatch.setattr(torch, 'load', forbidden)


def test_actual_forward_receiver_is_captured_twice_without_extra_inference(monkeypatch, tmp_path):
    import hyso.structure as structure
    class Tracked(nn.Sequential):
        calls = []
        def forward(self, inputs):
            self.calls.append((id(self), len(inputs)))
            return super().forward(inputs)
    class Once(Dataset):
        def __init__(self): self.seen = []
        def __len__(self): return 5
        def __getitem__(self, i):
            assert i not in self.seen
            self.seen.append(i)
            return torch.ones(3), i % 2
    model = Tracked(nn.Linear(3, 2)); dataset = Once(); seen = []; actual = structure.capture_model_structure
    def capture(receiver):
        count = len(Tracked.calls)
        with metadata_only(monkeypatch): result = actual(receiver)
        assert len(Tracked.calls) == count
        seen.append((id(receiver), result.content_sha256))
        return result
    monkeypatch.setattr(structure, 'capture_model_structure', capture)
    result = produce(model, DataLoader(dataset, batch_size=2)); d = result.to_dict(); b = d['structure_binding']
    assert d['schema_version'] == 3 and len(seen) == 2 and seen[0] == seen[1]
    assert seen[0][0] != id(model) and all(identity == seen[0][0] for identity, _ in Tracked.calls)
    assert [n for _, n in Tracked.calls] == [2, 2, 1] and dataset.seen == list(range(5))
    assert b['structure_sha256'] == b['pre_structure_sha256'] == b['post_structure_sha256'] == seen[0][1]
    assert b['evaluation_id'] == d['evaluation_id']
    assert b['model_state_sha256'] == b['pre_model_state_sha256'] == b['post_model_state_sha256'] == _model_hash(model)
    assert b['model_state_sha256'] == d['producer_evidence']['model_state']['sha256'] == d['source']['model_state_sha256']
    assert b['structure_sha256'] != b['model_state_sha256']
    assert b['forward_batches'] == 3 and b['evaluated_stream']['sample_count'] == 5
    assert b['evaluated_stream']['paired_sha256'] == d['producer_evidence']['evaluated_stream']['paired_sha256']
    assert b['checkpoint'] == dict(status='missing', sha256=None)
    assert result.structure_snapshot.to_dict() == b['snapshot']
    assert result.structure_snapshot.modules[0].configuration_status == 'unsupported'
    # Real serialization -> current reader -> native adapter -> resolver.
    r = record(tmp_path, result)
    reader_only(monkeypatch)
    reopened = EvaluationResult(json.loads((tmp_path / 'evaluation.json').read_text(encoding='utf-8')))
    assert reopened.to_dict() == d and reopened.structure_snapshot.content_sha256 == b['structure_sha256']
    native = resolver([r]); metric = native.resolve(r.record_id, M.TEST_ACCURACY).resolved
    assert native.check_metric(metric).status == 'supported'
    assert metric.reference.protocol_ref == 'hyso.evaluation.classification:schema3'
    context = metric.native_context['structure_binding']
    assert context['evaluation_id'] == d['evaluation_id'] and context['structure_sha256'] == b['structure_sha256']
    assert context['snapshot_path'] == 'structure_binding.snapshot' and context['forward_batches'] == 3
    assert metric.reference.record_id == r.record_id
    assert metric.reference.payload_sha256 == r.to_dict()['source']['payload_sha256']
    assert any('unsupported/missing' in note for note in metric.limitations)


def test_deepcopy_is_not_an_original_structure_or_behavior_equivalence_claim(monkeypatch):
    class Original(nn.Linear):
        def __deepcopy__(self, memo):
            other = nn.Linear(3, 2)
            with torch.no_grad(): other.weight.copy_(self.weight); other.bias.copy_(self.bias)
            other.extra_registered = nn.Identity()
            return other
        def forward(self, inputs): raise AssertionError('original must never run')
    model = Original(3, 2); result = produce(model); d = result.to_dict(); b = d['structure_binding']
    assert d['source']['model_type'].endswith('.Original')
    assert b['evaluated_model_type'] == 'torch.nn.modules.linear.Linear'
    assert b['snapshot']['modules'][0]['qualified_type'] == b['evaluated_model_type']
    assert result.structure_snapshot.summary.unique_modules == 2
    assert b['original_structure_relation'] == 'not_captured_or_equivalence_not_certified'
    assert b['model_state_sha256'] == _model_hash(model) and not hasattr(model, 'extra_registered')


def test_original_parameters_buffers_modes_gradients_rng_generator_and_hooks_preserved():
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Dropout(.3), nn.Linear(4, 2))
    model.train(); model[1].eval()
    for p in model.parameters(): p.grad = torch.ones_like(p)
    generator = torch.Generator().manual_seed(79); data = loader(); data.generator = generator
    def state():
        return (copy.deepcopy(model.state_dict()), [m.training for m in model.modules()],
                [p._version for p in model.parameters()], [p.grad.clone() for p in model.parameters()],
                [(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for m in model.modules()],
                generator.get_state().clone(), _rng_state())
    before = state(); result = produce(model, data)
    equal(before, state()); assert result.to_dict()['metrics']['sample_count'] == 5


def test_structure_changed_without_named_tensor_state_change_fails_closed():
    class Mutates(nn.Module):
        def __init__(self):
            super().__init__(); self.layer = nn.Linear(3, 2)
        def forward(self, x):
            self.layer.in_features += 1
            return self.layer(x)
    model = Mutates(); before = (copy.deepcopy(model.state_dict()), model.layer.in_features, _rng_state())
    with pytest.raises(ValueError, match='structure changed during evaluation'): produce(model)
    equal(before, (model.state_dict(), model.layer.in_features, _rng_state()))


def test_post_capture_state_hash_is_observed_not_assumed(monkeypatch):
    import hyso.structure as structure
    actual = structure.capture_model_structure; seen = []
    def changes_after_capture(receiver):
        result = actual(receiver); seen.append(receiver)
        if len(seen) == 2:
            with torch.no_grad(): receiver.weight.add_(1)
        return result
    monkeypatch.setattr(structure, 'capture_model_structure', changes_after_capture)
    model = nn.Linear(3, 2); before = (copy.deepcopy(model.state_dict()), _rng_state())
    with pytest.raises(ValueError, match='state changed during structure capture'): produce(model)
    equal(before, (model.state_dict(), _rng_state()))


@pytest.mark.parametrize('phase', ['pre', 'post'])
def test_capture_failures_restore_rng_and_generator(monkeypatch, phase):
    import hyso.structure as structure
    model = nn.Linear(3, 2); data = loader(); data.generator = torch.Generator().manual_seed(17)
    before = (copy.deepcopy(model.state_dict()), data.generator.get_state().clone(), _rng_state())
    actual = structure.capture_model_structure; calls = []
    def fail(receiver):
        calls.append(receiver)
        torch.rand(1)
        if len(calls) == (1 if phase == 'pre' else 2): raise RuntimeError('synthetic capture failure')
        return actual(receiver)
    monkeypatch.setattr(structure, 'capture_model_structure', fail)
    with pytest.raises(RuntimeError, match='synthetic capture failure'): produce(model, data)
    equal(before, (model.state_dict(), data.generator.get_state(), _rng_state()))


def test_checkpoint_receipt_separate_and_never_loaded_by_p26(tmp_path, monkeypatch):
    model = nn.Linear(3, 2); path = tmp_path / 'checkpoint.pth'
    save_training_checkpoint(path, model); blob = path.read_bytes()
    load_training_checkpoint(path, model, capture_evidence=True)
    def forbidden(*a, **kw): raise AssertionError('P26 must not load a checkpoint')
    monkeypatch.setattr(torch, 'load', forbidden)
    d = produce(model).to_dict(); b = d['structure_binding']
    assert b['checkpoint']['status'] == 'native' and b['checkpoint']['sha256'] == hashlib.sha256(blob).hexdigest()
    assert len({b['checkpoint']['sha256'], b['structure_sha256'], b['model_state_sha256']}) == 3
    assert b['checkpoint']['sha256'] == d['producer_evidence']['checkpoint']['sha256']
    assert path.read_bytes() == blob


def test_caller_declarations_do_not_create_or_repair_native_association(tmp_path):
    model = nn.Linear(3, 2)
    model.structure_sha256 = 'a'*64; model.checkpoint_sha256 = 'b'*64; model.model_id = 'trusted-looking-label'
    result = produce(model, source_id='caller-cohort')
    d = result.to_dict(); b = d['structure_binding']
    assert b['structure_sha256'] != model.structure_sha256 and b['checkpoint'] == dict(status='missing', sha256=None)
    r = record(tmp_path, result, scope=EvidenceScope(model_id='same-label', checkpoint_sha256='b'*64,
        dataset_id='caller-cohort', preprocessing_id='caller-transform', split='test'))
    native = resolver([r]); metric = native.resolve(r.record_id, M.TEST_MACRO_F1).resolved
    assert metric.native_context['structure_binding']['checkpoint']['status'] == 'missing'
    for key in ('model_id', 'checkpoint_sha256', 'dataset_id', 'preprocessing_id', 'split'):
        assert r.to_dict()['bindings'][key] == 'declared'
    assert r.to_dict()['bindings']['model_state_sha256'] == 'native'
    assert 'structure_sha256' not in r.to_dict()['scope']


@pytest.mark.parametrize('kwargs', [dict(capture_evidence=False), dict(capture_structure=1),
    dict(capture_structure=None), dict(capture_structure='native')])
def test_invalid_opt_in_rejected_before_model_execution(kwargs, monkeypatch):
    def forbidden(*a, **kw): raise AssertionError('invalid flags executed model')
    monkeypatch.setattr(nn.Module, '_call_impl', forbidden)
    with pytest.raises(ValueError): produce(**kwargs)


def test_empty_evaluation_cannot_claim_a_used_forward_receiver():
    data = DataLoader(TensorDataset(torch.empty(0, 3), torch.empty(0, dtype=torch.long)), batch_size=2)
    with pytest.raises(ValueError, match='nonempty evaluation'): produce(data=data)
    old = produce(data=data, capture_structure=False)
    assert old.to_dict()['schema_version'] == 2 and old.to_dict()['status'] == 'empty' and old.structure_snapshot is None


def previous_reader():
    source = subprocess.run(['git', 'show', '5c01b88cd8fc58250232514d5e10421c90d15439:hyso/evaluation.py'],
        cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True).stdout
    ns = {'__name__': 'hyso._previous_bound_eval_reader', '__package__': 'hyso'}
    exec(compile(source, '<trusted-pre-E1-reader>', 'exec'), ns)
    return ns


@pytest.mark.parametrize('evidence', [False, True])
def test_closed_feature_preserves_previous_schema_json_hashes_and_resolver_context(monkeypatch, tmp_path, evidence):
    import hyso.structure as structure
    old = previous_reader(); model = nn.Linear(3, 2); data = loader()
    monkeypatch.setattr(uuid, 'uuid4', lambda: uuid.UUID('10000000-0000-0000-0000-000000000000'))
    def forbidden(*a, **kw): raise AssertionError('closed feature captured structure')
    monkeypatch.setattr(structure, 'capture_model_structure', forbidden)
    options = dict(class_names=['a', 'b'], split='test', capture_evidence=evidence)
    previous = old['evaluate_classification'](model, data, **options).to_dict()
    current = evaluate_classification(model, data, **options)
    assert canonical(current.to_dict()) == canonical(previous)
    assert current.to_dict()['source_sha256'] == previous['source_sha256']
    assert current.to_dict()['schema_version'] == (2 if evidence else 1) and current.structure_snapshot is None
    r = record(tmp_path, current); metric = resolver([r]).resolve(r.record_id, M.TEST_ACCURACY).resolved
    assert 'structure_binding' not in metric.native_context


def test_old_reader_explicitly_rejects_new_schema_and_no_silent_downgrade():
    d = produce().to_dict()
    with pytest.raises(ValueError, match='invalid evaluation schema'): previous_reader()['EvaluationResult'](d)
    d['schema_version'] = 2
    with pytest.raises(ValueError, match='invalid evaluation schema'): EvaluationResult(d)


def rehash(data):
    data['source_sha256'] = _hash({k: v for k, v in data.items() if k not in ('source_sha256', 'metrics', 'status')})
    return data


@pytest.mark.parametrize('damage', ['missing', 'evaluation_id', 'structure_sha', 'post_structure', 'model_state',
    'pre_state', 'post_state', 'checkpoint', 'stream', 'stream_count', 'forward_batches', 'bool_batch',
    'model_type', 'phase', 'status', 'method', 'schema', 'scope', 'origin_claim', 'snapshot', 'torch', 'extra'])
def test_invalid_or_incoherent_binding_rejected_even_with_rehashed_native_envelope(damage):
    data = produce().to_dict(); b = data['structure_binding']
    if damage == 'missing': data.pop('structure_binding')
    elif damage == 'evaluation_id': b['evaluation_id'] = '20000000-0000-0000-0000-000000000000'
    elif damage == 'structure_sha': b['structure_sha256'] = 'a'*64
    elif damage == 'post_structure': b['post_structure_sha256'] = 'a'*64
    elif damage == 'model_state': b['model_state_sha256'] = b['structure_sha256']
    elif damage == 'pre_state': b['pre_model_state_sha256'] = 'a'*64
    elif damage == 'post_state': b['post_model_state_sha256'] = 'a'*64
    elif damage == 'checkpoint': b['checkpoint'] = dict(status='native', sha256='a'*64)
    elif damage == 'stream': b['evaluated_stream']['paired_sha256'] = 'a'*64
    elif damage == 'stream_count': b['evaluated_stream']['sample_count'] += 1
    elif damage == 'forward_batches': b['forward_batches'] += 1
    elif damage == 'bool_batch': b['forward_batches'] = True
    elif damage == 'model_type': b['evaluated_model_type'] = 'caller-declared-model'
    elif damage == 'phase': b['observed_phases'] = ['caller_snapshot']
    elif damage == 'status': b['status'] = 'declared'
    elif damage == 'method': b['method'] = 'caller_sha'
    elif damage == 'schema': b['schema_version'] = True
    elif damage == 'scope': b['scope'] = 'verified_execution_graph'
    elif damage == 'origin_claim': b['original_structure_relation'] = 'equivalent_because_deepcopy'
    elif damage == 'snapshot': b['snapshot']['summary']['parameter_elements']['value'] += 1
    elif damage == 'torch':
        b['snapshot']['producer']['torch_version'] = 'other-runtime'
        b['snapshot']['content_sha256'] = digest({k: v for k, v in b['snapshot'].items() if k != 'content_sha256'})
    else: b['native'] = True
    with pytest.raises(ValueError): EvaluationResult(rehash(data))


def test_native_adapter_rejects_broken_binding_before_it_can_be_resolved(tmp_path):
    data = produce().to_dict(); data['structure_binding']['status'] = 'missing'; rehash(data)
    path = tmp_path / 'bad.json'; path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(EvidenceError, match='invalid_native_evidence'):
        EvidenceRecord.from_json(path, feature_id='P26', expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize('task', ['binary', 'multiclass'])
def test_successful_binding_does_not_silently_extend_comparison_policy(tmp_path, task):
    records = []
    for i in range(2):
        data = loader()
        if task == 'binary': data = DataLoader(TensorDataset(data.dataset.tensors[0], data.dataset.tensors[1].float()), batch_size=2)
        result = produce(nn.Linear(3, 1 if task == 'binary' else 2), data, task=task)
        records.append(record(tmp_path, result, str(i)))
    native = resolver(records)
    for metric in (M.TEST_ACCURACY, M.TEST_MACRO_F1):
        assert native.resolve(records[0].record_id, metric).status == 'supported'
        for policy in (ComparisonContract(metric=metric, min_samples=1), P26QualityComparisonContract(metric=metric, allowed_changes=('model',))):
            comparison = compare_performance(native, records[0].record_id, native, records[1].record_id, contract=policy)
            assert comparison.eligibility.status == 'INSUFFICIENT_EVIDENCE' and comparison.difference is None


def test_schema3_csv_stays_metric_projection_and_snapshot_access_is_detached(tmp_path):
    result = produce(); before = result.to_dict()
    result.to_csv(tmp_path / 'metric.csv')
    detached = result.structure_snapshot.to_dict(); detached['modules'].clear()
    assert result.to_dict() == before and result.structure_snapshot.modules


@pytest.mark.parametrize('family', ['cnn', 'gru'])
def test_small_cpu_networks_bind_the_actual_evaluated_structure(family):
    if family == 'cnn':
        model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2, 2))
        inputs = torch.arange(125, dtype=torch.float32).reshape(5, 1, 5, 5) / 100
    else:
        class Classifier(nn.Module):
            def __init__(self):
                super().__init__(); self.gru = nn.GRU(3, 4, batch_first=True); self.head = nn.Linear(4, 2)
            def forward(self, x):
                sequence, _ = self.gru(x)
                return self.head(sequence[:, -1, :])
        model = Classifier()
        inputs = torch.arange(30, dtype=torch.float32).reshape(5, 2, 3) / 100
    data = DataLoader(TensorDataset(inputs, torch.tensor([0, 1, 0, 1, 0])), batch_size=2)
    result = produce(model, data)
    assert result.to_dict()['schema_version'] == 3 and result.to_dict()['metrics']['sample_count'] == 5
    categories = {c.name: c.unique_modules for c in result.structure_snapshot.summary.categories}
    assert categories['conv2d' if family == 'cnn' else 'recurrent'] == 1
