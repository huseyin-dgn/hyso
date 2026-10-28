"""3B.4C.1: actual P26 producer, synthetic CPU inference, no training/benchmark."""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, default_collate

from hyso import EvaluationResult, evaluate_classification, evaluate_predictions
from hyso._evaluation_evidence import EvaluationStreamCapture
from hyso.analysis import AnalysisContext, EvidenceIndex, EvidenceRecord, EvidenceScope
from hyso.analysis._codec import canonical
from hyso.advisor import (ComparisonContract, NativeMetric, NativeMetricResolver,
                          compare_performance, compare_records)
from hyso.checkpoint import load_training_checkpoint, save_training_checkpoint, _rng_state
from hyso.evaluation import _hash, _model_hash
from trainer_replay_support import equal


@pytest.fixture(autouse=True)
def no_device_training_or_benchmark(monkeypatch):
    from hyso.benchmark import BenchmarkRunner
    from hyso.training import Trainer
    def forbidden(*args, **kwargs):
        raise AssertionError("training/benchmark/GPU calls forbidden")
    for name in ("init", "is_available", "device_count", "current_device"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(Trainer, "fit", forbidden)
    monkeypatch.setattr(BenchmarkRunner, "run", forbidden)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


def samples():
    return torch.arange(15, dtype=torch.float32).reshape(5, 3) / 20, torch.tensor([0, 1, 0, 1, 0])


def produce(model=None, x=None, y=None, batch_size=2, **kwargs):
    original_x, original_y = samples()
    data = TensorDataset(original_x if x is None else x, original_y if y is None else y)
    return evaluate_classification(nn.Linear(3, 2) if model is None else model,
        DataLoader(data, batch_size=batch_size), class_names=["a", "b"], split="test",
        capture_evidence=True, **kwargs)


def stream(result):
    return result.to_dict()["producer_evidence"]["evaluated_stream"]


def import_result(tmp_path, result, name="capture", scope=None):
    path = tmp_path / (name + ".json")
    result.to_json(path)
    return EvidenceRecord.from_json(path, feature_id="P26", scope=scope,
                                   expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_real_producer_state_identity_and_model_rng_isolation(tmp_path):
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Dropout(.3), nn.Linear(4, 2))
    model.train(); model[1].eval()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    before = (copy.deepcopy(model.state_dict()), [m.training for m in model.modules()],
              [p.grad.clone() for p in model.parameters()], [p._version for p in model.parameters()], _rng_state())
    # Independent historical model hash algorithm, including default JSON spaces.
    expected = hashlib.sha256()
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        expected.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        expected.update(tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    result = produce(model)
    d = result.to_dict(); evidence = d["producer_evidence"]
    assert d["schema_version"] == 2
    assert evidence["model_state"]["sha256"] == d["source"]["model_state_sha256"] == expected.hexdigest()
    equal(before, (model.state_dict(), [m.training for m in model.modules()],
                  [p.grad for p in model.parameters()], [p._version for p in model.parameters()], _rng_state()))
    record = import_result(tmp_path, result).to_dict()
    assert record["scope"]["model_state_sha256"] == expected.hexdigest()
    assert record["bindings"]["model_state_sha256"] == "native"
    assert d["accounting"] == dict(expected_samples=5, batch_sizes=[2, 2, 1])
    assert stream(result)["sample_count"] == 5 == d["metrics"]["sample_count"]


def test_deterministic_stream_identity_independent_of_batch_partition():
    model = nn.Linear(3, 2)
    results = [produce(model, batch_size=n) for n in (1, 2, 3, 5)]
    assert all(stream(r) == stream(results[0]) for r in results)
    assert stream(results[0])["batch_partition_invariant"] is True
    assert stream(results[0])["scope"] == "ordered_post_collation_inputs_and_targets"


def test_stream_v1_fixed_vector_from_independent_struct_encoding():
    # Frozen from the documented framing using stdlib struct.pack('<3f') and
    # struct.pack('<q'), without Torch or the production hashing implementation.
    result = produce(x=torch.tensor([[1., 2., 3.], [4., 5., 6.]]), y=torch.tensor([0, 1]))
    evidence = stream(result)
    assert evidence["inputs_sha256"] == "8d936b7190b0402d760e4336c69f494a8bd3183e1e49a3029d00190507b106a6"
    assert evidence["targets_sha256"] == "bf76d11b373ac857d2d6ff168c6611acbbbaf2cbc178365619d86ba08db75b89"
    assert evidence["paired_sha256"] == "b643ade7b8fc24b82275f80af958d816768178b573fd66b4228f6771c0695536"


def test_changed_deepcopy_cannot_claim_original_model_identity():
    class ChangedCopy(nn.Linear):
        def __deepcopy__(self, memo):
            other = nn.Linear(3, 2)
            with torch.no_grad():
                other.weight.copy_(self.weight + 1)
                other.bias.copy_(self.bias)
            return other
    model = ChangedCopy(3, 2)
    before = (copy.deepcopy(model.state_dict()), _rng_state())
    with pytest.raises(ValueError, match="isolated model state differs from original"):
        produce(model)
    equal(before, (model.state_dict(), _rng_state()))


@pytest.mark.parametrize("change", ["input", "label", "order", "label_order", "dtype", "shape"])
def test_same_shape_and_class_do_not_prove_same_content(change):
    x, y = samples(); original = stream(produce(x=x, y=y))
    if change == "input":
        x[0, 0] += .125
    elif change == "label":
        y[0] = 1
    elif change == "order":
        x, y = x.flip(0), y.flip(0)
    elif change == "label_order":
        y = y.roll(1)
    elif change == "dtype":
        x = x.double()
    elif change == "shape":
        x = x.reshape(5, 1, 3)
    model = nn.Sequential(nn.Flatten(1), nn.Linear(3, 2))
    if change == "dtype":
        model.double()
    changed = stream(produce(model, x=x, y=y))
    assert changed["paired_sha256"] != original["paired_sha256"]
    if change in ("input", "order", "dtype", "shape"):
        assert changed["inputs_sha256"] != original["inputs_sha256"]
    else:
        assert changed["inputs_sha256"] == original["inputs_sha256"]
        assert changed["targets_sha256"] != original["targets_sha256"]


def test_capture_uses_only_existing_iteration_and_post_collation_content():
    class Once(Dataset):
        def __init__(self):
            self.seen = []
        def __len__(self):
            return 5
        def __getitem__(self, index):
            assert index not in self.seen, "dataset read twice"
            self.seen.append(index)
            return samples()[0][index], samples()[1][index]
    dataset = Once()
    def transform(items):
        x, y = default_collate(items)
        return x + 7, 1-y
    result = evaluate_classification(nn.Linear(3, 2), DataLoader(dataset, batch_size=2, collate_fn=transform),
                                    class_names=["a", "b"], split="test", capture_evidence=True)
    x, y = samples()
    assert dataset.seen == list(range(5))
    assert stream(result) == stream(produce(x=x+7, y=1-y))
    evidence = result.to_dict()["producer_evidence"]
    assert evidence["preprocessing"] == dict(status="missing", reason="transform_chain_not_verified")
    assert set(evidence["evaluated_stream"]) == {
        "status", "method", "scope", "source", "sample_count", "batch_partition_invariant",
        "inputs_sha256", "targets_sha256", "paired_sha256"}


def test_no_checkpoint_and_declarations_never_promoted(tmp_path):
    model = nn.Linear(3, 2)
    model.checkpoint_sha256 = "a"*64
    model.checkpoint_path = "caller-checkpoint.pth"
    model.preprocessing = "trusted-looking callable name"
    result = produce(model, source_id="caller-cohort")
    evidence = result.to_dict()["producer_evidence"]
    assert evidence["checkpoint"]["status"] == "missing"
    assert evidence["checkpoint"]["sha256"] is None
    assert evidence["split"]["status"] == evidence["source_id"]["status"] == "declared"
    assert evidence["dataset_origin"]["status"] == evidence["preprocessing"]["status"] == "missing"
    record = import_result(tmp_path, result, scope=EvidenceScope(checkpoint_sha256="a"*64,
        preprocessing_id="external-transform", dataset_id="caller-cohort", split="test")).to_dict()
    for field in ("checkpoint_sha256", "preprocessing_id", "dataset_id", "split"):
        assert record["bindings"][field] == "declared"
    missing = import_result(tmp_path, result, "missing").to_dict()
    for field in ("checkpoint_sha256", "preprocessing_id", "dataset_id"):
        assert missing["bindings"][field] == "missing"
    assert missing["bindings"]["split"] == "declared"
    with pytest.raises(ValueError, match="declared_split_mismatch"):
        import_result(tmp_path, result, "bad-split", EvidenceScope(split="train"))


def test_trusted_checkpoint_load_binds_exact_artifact_without_changing_file(tmp_path):
    original, loaded = nn.Linear(3, 2), nn.Linear(3, 2)
    path = tmp_path / "checkpoint.pth"
    save_training_checkpoint(path, original)
    before = path.read_bytes()
    load_training_checkpoint(path, loaded, capture_evidence=True)
    result = produce(loaded)
    checkpoint = result.to_dict()["producer_evidence"]["checkpoint"]
    assert checkpoint["status"] == "native"
    assert checkpoint["sha256"] == hashlib.sha256(before).hexdigest()
    assert checkpoint["model_state_sha256"] == _model_hash(loaded)
    assert checkpoint["sha256"] != checkpoint["model_state_sha256"]
    assert path.read_bytes() == before
    record = import_result(tmp_path, result).to_dict()
    assert record["bindings"]["checkpoint_sha256"] == "native"
    assert record["scope"]["checkpoint_sha256"] == checkpoint["sha256"]
    # A copy or caller-reconstructed identical model has no load receipt.
    assert produce(copy.deepcopy(loaded)).to_dict()["producer_evidence"]["checkpoint"]["status"] == "missing"
    with torch.no_grad():
        loaded.weight.add_(1)
    assert produce(loaded).to_dict()["producer_evidence"]["checkpoint"]["status"] == "missing"


def test_checkpoint_snapshot_not_a_second_path_read_and_stream_cursor_restored(tmp_path, monkeypatch):
    model = nn.Linear(3, 2); path = tmp_path / "state.pth"
    save_training_checkpoint(path, model)
    blob = path.read_bytes(); original_load = torch.load
    def replace_path_during_load(source, **kwargs):
        path.write_bytes(b"replaced after snapshot")
        assert kwargs["weights_only"] is True
        return original_load(source, **kwargs)
    monkeypatch.setattr(torch, "load", replace_path_during_load)
    target = nn.Linear(3, 2)
    load_training_checkpoint(path, target, capture_evidence=True)
    assert produce(target).to_dict()["producer_evidence"]["checkpoint"]["sha256"] == hashlib.sha256(blob).hexdigest()
    source = io.BytesIO(blob)
    load_training_checkpoint(source, target, capture_evidence=True)
    assert source.tell() == 0
    assert produce(target).to_dict()["producer_evidence"]["checkpoint"]["status"] == "native"


def test_uncaptured_reload_clears_receipt_and_failed_restore_never_mints_one(tmp_path):
    from hyso.checkpoint import TrainingCheckpointError
    path = tmp_path / "state.pth"; model = nn.Linear(3, 2)
    save_training_checkpoint(path, model)
    load_training_checkpoint(path, model, capture_evidence=True)
    load_training_checkpoint(path, model)
    assert produce(model).to_dict()["producer_evidence"]["checkpoint"]["status"] == "missing"
    class Fails(nn.Linear):
        def load_state_dict(self, state, *args, **kwargs):
            raise ValueError("restore failure")
    failing = Fails(3, 2); save_training_checkpoint(path, failing)
    with pytest.raises(TrainingCheckpointError):
        load_training_checkpoint(path, failing, capture_evidence=True)
    assert produce(failing).to_dict()["producer_evidence"]["checkpoint"]["status"] == "missing"


@pytest.mark.parametrize("kind", ["nonpersistent_buffer", "changed_state", "mutated_source"])
def test_partial_or_custom_restore_does_not_claim_full_state_binding(tmp_path, kind):
    class Partial(nn.Linear):
        def __init__(self):
            super().__init__(3, 2)
            if kind == "nonpersistent_buffer":
                self.register_buffer("outside_checkpoint", torch.ones(1), persistent=False)
        def load_state_dict(self, state, *args, **kwargs):
            if kind == "mutated_source":
                # A load hook can mutate the deserialized payload as well as the
                # target. Expected bytes must have been frozen before that hook.
                state["weight"].add_(1)
            result = super().load_state_dict(state, *args, **kwargs)
            if kind == "changed_state":
                with torch.no_grad():
                    self.weight.neg_()  # Includes the +0/-0 distinction.
            return result
    model = Partial()
    with torch.no_grad(): model.weight.zero_()
    path = tmp_path / "partial.pth"; save_training_checkpoint(path, model)
    load_training_checkpoint(path, model, capture_evidence=True)
    assert produce(model).to_dict()["producer_evidence"]["checkpoint"]["status"] == "missing"


def test_captured_load_never_retries_unsafe_pickle(tmp_path, monkeypatch):
    from hyso.checkpoint import TrainingCheckpointError
    path = tmp_path / "unsafe.pth"; model = nn.Linear(3, 2)
    torch.save(model, path)
    original = torch.load; calls = []
    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "load", checked_load)
    with pytest.raises(TrainingCheckpointError, match="weights-only"):
        load_training_checkpoint(path, model, capture_evidence=True)
    assert len(calls) == 1 and calls[0]["weights_only"] is True
    assert produce(model).to_dict()["producer_evidence"]["checkpoint"]["status"] == "missing"


def test_capture_failure_restores_loader_generator_and_rng(monkeypatch):
    generator = torch.Generator().manual_seed(73)
    x, y = samples(); data = DataLoader(TensorDataset(x, y), batch_size=2, generator=generator)
    model = nn.Linear(3, 2)
    before = (copy.deepcopy(model.state_dict()), generator.get_state().clone(), _rng_state())
    def fail(*args):
        torch.rand(1)
        raise RuntimeError("hash capture failed")
    monkeypatch.setattr(EvaluationStreamCapture, "update", fail)
    with pytest.raises(RuntimeError, match="hash capture failed"):
        evaluate_classification(model, data, class_names=["a", "b"], split="test", capture_evidence=True)
    equal(before, (model.state_dict(), generator.get_state(), _rng_state()))


def test_default_producer_remains_schema1_without_content_hash_work(monkeypatch):
    def forbidden(*args):
        raise AssertionError("default mode must not collect content")
    monkeypatch.setattr(EvaluationStreamCapture, "update", forbidden)
    x, y = samples()
    result = evaluate_classification(nn.Linear(3, 2), DataLoader(TensorDataset(x, y), batch_size=2),
                                    class_names=["a", "b"], split="test")
    assert result.to_dict()["schema_version"] == 1
    assert "producer_evidence" not in result.to_dict()
    for value in (1, None, "v1"):
        with pytest.raises(ValueError, match="must be bool"):
            evaluate_classification(
                nn.Linear(3, 2), DataLoader(TensorDataset(x, y)), class_names=["a", "b"],
                split="test", capture_evidence=value)


@pytest.mark.parametrize("damage", ["count", "method", "scope", "split", "preprocessing", "checkpoint", "hash", "version", "extra"])
def test_new_reader_rejects_incoherent_schema_even_with_rehashed_envelope(damage):
    data = produce().to_dict(); evidence = data["producer_evidence"]
    if damage == "count": evidence["evaluated_stream"]["sample_count"] = 4
    elif damage == "method": evidence["evaluated_stream"]["method"] = "repr"
    elif damage == "scope": evidence["evaluated_stream"]["scope"] = "dataset_origin"
    elif damage == "split": evidence["split"]["status"] = "native"
    elif damage == "preprocessing": evidence["preprocessing"]["status"] = "native"
    elif damage == "checkpoint": evidence["checkpoint"]["status"] = "native"
    elif damage == "hash": evidence["evaluated_stream"]["inputs_sha256"] = "z"*64
    elif damage == "version": evidence["schema_version"] = True
    else: evidence["native"] = True
    data["source_sha256"] = _hash({k: v for k, v in data.items() if k not in ("source_sha256", "metrics", "status")})
    with pytest.raises(ValueError): EvaluationResult(data)


def test_new_payload_cannot_downgrade_or_upgrade_supplied_predictions():
    data = produce().to_dict(); data["schema_version"] = 1
    with pytest.raises(ValueError): EvaluationResult(data)
    supplied = evaluate_predictions([0], [0], class_names=["a", "b"], split="test").to_dict()
    supplied.update(schema_version=2, producer_evidence=produce().to_dict()["producer_evidence"])
    with pytest.raises(ValueError): EvaluationResult(supplied)


def test_reader_roundtrip_and_advisor_comparison_remain_fail_closed(tmp_path):
    records = [import_result(tmp_path, produce(), str(i)) for i in range(2)]
    index = EvidenceIndex(records)
    report = index.analyze(context=AnalysisContext(workload_id="synthetic-p26", task="image_classification", model={}, dataset={}))
    before = canonical(report.to_dict())
    resolver = NativeMetricResolver(report)
    metric = resolver.resolve(records[0].record_id, NativeMetric.TEST_ACCURACY).resolved
    assert metric.reference.protocol_ref == "hyso.evaluation.classification:schema2"
    assert resolver.check_metric(metric).status == "supported"
    assert any("caller-declared" in note for note in metric.limitations)
    policy = ComparisonContract(metric=NativeMetric.TEST_ACCURACY, min_samples=1)
    compared = compare_records(resolver, records[0].record_id, resolver, records[1].record_id, contract=policy)
    assert compared.status == "INSUFFICIENT_EVIDENCE"
    performance = compare_performance(resolver, records[0].record_id, resolver, records[1].record_id, contract=policy)
    assert performance.eligibility.status == "INSUFFICIENT_EVIDENCE" and performance.difference is None
    assert canonical(report.to_dict()) == before
    index.to_json(tmp_path / "index.json")
    assert EvidenceIndex.from_json(tmp_path / "index.json").to_dict() == index.to_dict()
    result = produce(); result.to_csv(tmp_path / "new.csv")
    assert EvaluationResult(result.to_dict()).to_dict() == result.to_dict()


def test_empty_binary_noncontiguous_and_failure_paths():
    empty = produce(x=torch.empty(0, 3), y=torch.empty(0, dtype=torch.long))
    assert stream(empty)["sample_count"] == 0 and empty.to_dict()["status"] == "empty"
    x, _ = samples()
    # Identical logical samples with different physical strides hash identically.
    noncontiguous = x.t().contiguous().t()
    assert not noncontiguous.is_contiguous()
    assert stream(produce(x=x)) == stream(produce(x=noncontiguous))
    binary = evaluate_classification(nn.Linear(3, 1), DataLoader(TensorDataset(x, torch.ones(5, 1)), batch_size=2),
        class_names=["no", "yes"], task="binary", split="test", capture_evidence=True)
    assert stream(binary)["sample_count"] == 5
    class Failure(nn.Linear):
        def forward(self, x):
            torch.rand(1)
            raise RuntimeError("synthetic failure")
    model = Failure(3, 2); before = (copy.deepcopy(model.state_dict()), _rng_state())
    with pytest.raises(RuntimeError, match="synthetic failure"):
        produce(model)
    equal(before, (model.state_dict(), _rng_state()))


def test_previous_reader_explicitly_rejects_new_schema_accepts_default_schema(monkeypatch, tmp_path):
    # Execute only the trusted previous reader definition; never any model source.
    repo = Path(__file__).resolve().parents[1]
    source = subprocess.run(["git", "show", "075906dff661abcc26df921a9cd6b455ae7db2f9:hyso/evaluation.py"],
                            cwd=repo, check=True, capture_output=True, text=True).stdout
    namespace = {"__name__": "hyso._previous_p26_reader", "__package__": "hyso"}
    exec(compile(source, "<previous-trusted-p26-reader>", "exec"), namespace)
    with pytest.raises(ValueError, match="invalid evaluation schema"):
        namespace["EvaluationResult"](produce().to_dict())
    old = evaluate_predictions([0, 1], [0, 1], class_names=["a", "b"], split="test").to_dict()
    assert namespace["EvaluationResult"](old).to_dict() == old
    # Fixed UUID allows exact old/current producer JSON and source hash equality.
    import uuid
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID("10000000-0000-0000-0000-000000000000"))
    x, y = samples(); model = nn.Linear(3, 2); data = DataLoader(TensorDataset(x, y), batch_size=2)
    kwargs = dict(class_names=["a", "b"], split="test")
    previous = namespace["evaluate_classification"](model, data, **kwargs).to_dict()
    current = evaluate_classification(model, data, **kwargs)
    assert canonical(previous) == canonical(current.to_dict())
    assert previous["source_sha256"] == current.to_dict()["source_sha256"]
    record = import_result(tmp_path, EvaluationResult(previous))
    assert EvidenceRecord(record.to_dict()).to_dict() == record.to_dict()
    assert record.to_dict()["native_payload"] == previous
    assert "producer_evidence" not in record.to_dict()["native_payload"]
    assert record.to_dict()["bindings"]["checkpoint_sha256"] == "missing"


def test_existing_mnist_reports_and_evidence_are_unchanged():
    # Optional local retained acceptance artifacts, never regenerated/migrated.
    root = Path(__file__).resolve().parents[1] / "outputs/advisor-reporting-3b3c-20261028"
    paths = sorted(root.glob("mnist-*.display.json"))
    if not paths:
        pytest.skip("retained local MNIST acceptance artifacts unavailable")
    from hyso.advisor import AdvisorReport
    for path in paths:
        raw = path.read_bytes(); view = json.loads(raw)
        old = view["data"]["original_technical_record"]
        report = AdvisorReport.from_dict(old)
        assert report.to_dict() == old
        assert report.report_id == view["source_advisor_report_id"]
        language, audience = path.name.removesuffix(".display.json").split("-")[1:3]
        assert report.to_display_dict(language=language, audience=audience) == view
        assert report.to_markdown(language=language, audience=audience) == path.with_name(path.name.replace(".display.json", ".md")).read_text(encoding="utf-8")
        assert path.read_bytes() == raw
