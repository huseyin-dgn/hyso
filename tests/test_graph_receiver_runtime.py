"""Actual evaluator receiver authority, separate from checkpoint/P26 statements."""

import copy
import json

import pytest
import torch

from hyso import evaluate_classification
from hyso.analysis.graph_evidence import (
    GraphEvaluationResult,
    evaluate_graph_classification,
    compare_graph_evaluations,
)
from hyso.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
    graph_checkpoint_evidence,
)
from hyso.reporting import create_report
from test_graph_checkpoint_contract import model
from test_graph_evidence_integration import loader


def trusted(tmp_path, name="a", dtype=torch.float32):
    source = model().to(dtype).eval()
    path = tmp_path / (name + ".pth")
    save_training_checkpoint(path, source, graph_contract=True, include_rng=False)
    load_training_checkpoint(path, source, capture_evidence=True)
    return source


def run(source, data=None, **kwargs):
    return evaluate_graph_classification(
        source,
        loader() if data is None else data,
        class_names=["a", "b"],
        split="test",
        source_id="receiver-fixture",
        context=dict(
            dataset_id="receiver-fixture", split="test", preprocessing_id="identity"
        ),
        capture_receiver_runtime=True,
        **kwargs,
    )


@pytest.mark.parametrize("structure", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_actual_receiver_runtime_is_verified_with_unchanged_p26(
    tmp_path, structure, dtype
):
    source = trusted(tmp_path, dtype=dtype)
    result = run(source, loader(dtype=dtype), capture_structure=structure)
    data = result.to_dict()
    assert data["schema_version"] == 2
    assert data["verification"]["graph_v2"] == "verified"
    assert data["verification"]["receiver_runtime"] == "verified"
    assert data["relation"]["status"] == "verified"
    assert data["relation"]["isolated_receiver_receipt"] == "not_transferred"
    assert data["evaluation"]["schema_version"] == (3 if structure else 2)
    assert data["evaluation"]["producer_evidence"]["checkpoint"]["status"] == "missing"
    binding = data["receiver_binding"]
    assert binding["source_object_token"] != binding["receiver_object_token"]
    assert binding["forward_batches"] == 2 and binding["sample_count"] == 4
    assert binding["pre_contract_sha256"] == binding["post_contract_sha256"]
    assert binding["receiver_contract"] == data["source_receipt"]["contract"]


def test_default_graph_association_remains_legacy_limited(tmp_path):
    source = trusted(tmp_path)
    old = evaluate_graph_classification(
        source, loader(), class_names=["a", "b"], split="test"
    )
    assert old.to_dict()["schema_version"] == 1
    assert old.to_dict()["relation"]["status"] == "limited"


@pytest.mark.parametrize(
    "change",
    [
        "parameter",
        "persistent",
        "activation",
        "topology",
        "output",
        "geometry",
        "dtype",
        "nonpersistent",
    ],
)
def test_source_mutation_cannot_verify_receiver(tmp_path, change):
    source = trusted(tmp_path)
    if change == "parameter":
        with torch.no_grad():
            next(source.parameters()).add_(1)
    elif change == "persistent":
        next(v for n, v in source.named_buffers() if n.endswith("running_mean")).add_(1)
    elif change == "activation":
        source.graph.nodes["node_stem"].activation = torch.nn.ReLU()
    elif change == "output":
        source.output_name = "features"
    elif change == "dtype":
        source.double()
    elif change == "nonpersistent":
        source.graph._contract = torch.zeros(1)
    else:
        config = source.graph.to_config()
        if change == "geometry":
            config["inputs"]["image"]["shape"] = [1, 5, 5]
        else:
            next(r for r in config["nodes"] if r["id"] == "merge")["component"] = (
                "multiply"
            )
        source.graph._architecture = json.dumps(config, sort_keys=True)
    with pytest.raises((ValueError, NotImplementedError)):
        run(source)


@pytest.mark.parametrize("operation", ["json", "constructor", "deepcopy"])
def test_retained_runtime_claim_never_has_live_authority(tmp_path, operation):
    source = trusted(tmp_path)
    result = run(source)
    if operation == "deepcopy":
        imported = copy.deepcopy(result)
    elif operation == "constructor":
        imported = GraphEvaluationResult(result.to_dict())
    else:
        path = tmp_path / "result.json"
        result.to_json(path)
        imported = GraphEvaluationResult.from_json(path)
    assert imported.to_dict()["verification"]["receiver_runtime"] == "unverified"
    assert imported.to_dict()["relation"]["status"] != "verified"


def test_comparison_reporting_show_each_binding_without_p26_upgrade(tmp_path):
    a, b = trusted(tmp_path, "a"), trusted(tmp_path, "b")
    left, right = run(a), run(b)
    compared = compare_graph_evaluations(left, right)
    assert compared.to_dict()["status"] == "COMPARABLE"
    assert all(
        v["inference_binding"] == "verified"
        for v in compared.to_dict()["verification"].values()
    )
    report = create_report(left, right, compared, language="en")
    assert (
        report.to_dict()["sources"][0]["graph_evidence"]["cross_evidence_binding"]
        == "verified"
    )
    assert "verified" in report.to_markdown() and "verified" in report.to_html()
    with torch.no_grad():
        next(a.parameters()).add_(1)
    assert left.to_dict()["relation"]["status"] != "verified"
    assert left.to_dict()["verification"]["graph_v2"] == "missing"


def test_core_requires_explicit_evidence_and_finite_nonempty_receiver_execution(
    tmp_path,
):
    source = trusted(tmp_path)
    with pytest.raises(ValueError):
        evaluate_classification(
            source,
            loader(),
            class_names=["a", "b"],
            split="test",
            capture_receiver_runtime=True,
        )
    empty = loader()
    empty.dataset.tensors = tuple(t[:0] for t in empty.dataset.tensors)
    with pytest.raises(ValueError):
        run(source, empty)


def test_deepcopy_and_another_source_never_transfer_checkpoint_receipt(tmp_path):
    source = trusted(tmp_path)
    clone = copy.deepcopy(source)
    assert graph_checkpoint_evidence(clone)["status"] == "missing"
    with pytest.raises(ValueError):
        run(clone)


def observe_receiver(monkeypatch):
    import weakref
    from hyso._graph_receiver import _ReceiverCapture

    actual = _ReceiverCapture.start
    observed = []

    def start(capture, receiver):
        observed.append(weakref.ref(receiver))
        return actual(capture, receiver)

    monkeypatch.setattr(_ReceiverCapture, "start", start)
    return observed


def mutate(receiver, change):
    if change == "parameter":
        with torch.no_grad():
            next(receiver.parameters()).add_(1)
    elif change == "persistent":
        next(v for n, v in receiver.named_buffers() if n.endswith("running_mean")).add_(
            1
        )
    elif change == "activation":
        receiver.graph.nodes["node_stem"].activation = torch.nn.ReLU()
    elif change == "output":
        receiver.output_name = "features"
    elif change == "geometry":
        config = receiver.graph.to_config()
        config["inputs"]["image"]["shape"] = [1, 5, 5]
        receiver.graph._architecture = json.dumps(config, sort_keys=True)
    elif change == "topology":
        config = receiver.graph.to_config()
        next(row for row in config["nodes"] if row["id"] == "merge")["component"] = (
            "multiply"
        )
        receiver.graph._architecture = json.dumps(config, sort_keys=True)
    elif change == "nonpersistent":
        receiver.graph._contract = torch.ones(1)
    elif change == "dtype":
        receiver.double()
    elif change == "device":
        receiver.graph._contract = torch.empty(0, device="meta")
    else:
        raise ValueError(change)


@pytest.mark.parametrize(
    "change",
    [
        "parameter",
        "persistent",
        "activation",
        "topology",
        "output",
        "geometry",
        "nonpersistent",
        "dtype",
        "device",
    ],
)
def test_actual_receiver_mutation_before_forward_is_rejected(
    tmp_path, monkeypatch, change
):
    from torch.utils.data import Dataset, DataLoader

    source = trusted(tmp_path)
    observed = observe_receiver(monkeypatch)
    tensors = loader().dataset

    class Changing(Dataset):
        def __len__(self):
            return len(tensors)

        def __getitem__(self, index):
            if index == 0:
                mutate(observed[0](), change)
            return tensors[index]

    before = graph_checkpoint_evidence(source)
    with pytest.raises((ValueError, NotImplementedError)):
        run(source, DataLoader(Changing(), batch_size=2))
    assert graph_checkpoint_evidence(source) == before


@pytest.mark.parametrize(
    "change",
    ["parameter", "persistent", "activation", "topology", "output", "nonpersistent"],
)
def test_actual_receiver_mutation_after_forward_prevents_publication(
    tmp_path, monkeypatch, change
):
    from hyso._graph_receiver import _ReceiverCapture

    source = trusted(tmp_path)
    actual = _ReceiverCapture.after
    calls = []

    def after(capture, receiver, logits):
        calls.append(id(receiver))
        mutate(receiver, change)
        return actual(capture, receiver, logits)

    monkeypatch.setattr(_ReceiverCapture, "after", after)
    with pytest.raises(ValueError):
        run(source)
    assert len(calls) == 1


def test_same_actual_object_brackets_all_forward_calls_and_has_no_checkpoint_receipt(
    tmp_path, monkeypatch
):
    from hyso._graph_receiver import _ReceiverCapture
    from hyso._evaluation_evidence import model_hash

    source = trusted(tmp_path)
    observed = observe_receiver(monkeypatch)
    before, after = _ReceiverCapture.before, _ReceiverCapture.after
    calls = []

    def pre(capture, receiver, *args):
        calls.append(("before", id(receiver)))
        return before(capture, receiver, *args)

    def post(capture, receiver, *args):
        calls.append(("after", id(receiver)))
        return after(capture, receiver, *args)

    monkeypatch.setattr(_ReceiverCapture, "before", pre)
    monkeypatch.setattr(_ReceiverCapture, "after", post)
    result = run(source)
    receiver = observed[0]()
    assert receiver is not None and receiver is not source
    assert [phase for phase, _ in calls] == ["before", "after", "before", "after"]
    assert all(identity == id(receiver) for _, identity in calls)
    assert model_hash(source) == model_hash(receiver)
    assert graph_checkpoint_evidence(receiver)["status"] == "missing"
    assert result.to_dict()["relation"]["status"] == "verified"


def test_wrong_receiver_object_never_satisfies_observation(tmp_path, monkeypatch):
    from hyso._graph_receiver import _ReceiverCapture

    source = trusted(tmp_path)
    other = copy.deepcopy(source)
    actual = _ReceiverCapture.before

    def replaced(capture, receiver, *args):
        return actual(capture, other, *args)

    monkeypatch.setattr(_ReceiverCapture, "before", replaced)
    with pytest.raises(ValueError, match="object identity"):
        run(source)


@pytest.mark.parametrize("change", ["parameter", "activation", "nonpersistent"])
def test_post_evaluation_receiver_mutation_downgrades_live_view(
    tmp_path, monkeypatch, change
):
    source = trusted(tmp_path)
    observed = observe_receiver(monkeypatch)
    result = run(source)
    assert result.to_dict()["relation"]["status"] == "verified"
    mutate(observed[0](), change)
    data = result.to_dict()
    assert data["verification"]["graph_v2"] == "verified"
    assert data["verification"]["receiver_runtime"] == "missing"
    assert data["relation"]["status"] != "verified"


@pytest.mark.parametrize(
    "kind",
    ["local_hook", "global_hook", "subclass", "instance_forward", "class_forward"],
)
def test_unsupported_runtime_never_verifies(tmp_path, monkeypatch, kind):
    from hyso.models.output import GraphTensorOutput

    source = trusted(tmp_path)
    handle = None
    if kind == "local_hook":
        handle = source.register_forward_hook(lambda *args: None)
    elif kind == "global_hook":
        handle = torch.nn.modules.module.register_module_forward_hook(
            lambda *args: None
        )
    elif kind == "subclass":

        class Custom(GraphTensorOutput):
            pass

        source.__class__ = Custom
    elif kind == "instance_forward":
        source.forward = lambda x: torch.zeros(len(x), 2)
    else:
        monkeypatch.setattr(torch.nn.SiLU, "forward", lambda self, x: -x)
    try:
        with pytest.raises(ValueError):
            run(source)
    finally:
        if handle is not None:
            handle.remove()


@pytest.mark.parametrize("phase", ["dataset", "forward"])
def test_failed_or_partial_inference_has_no_published_binding(
    tmp_path, monkeypatch, phase
):
    from hyso._graph_receiver import _ReceiverCapture, _RESULTS

    source = trusted(tmp_path)
    before = set(_RESULTS)
    if phase == "forward":
        actual = _ReceiverCapture.after

        def fail(capture, receiver, logits):
            actual(capture, receiver, logits)
            raise RuntimeError("explicit isolated forward failure")

        monkeypatch.setattr(_ReceiverCapture, "after", fail)
        with pytest.raises(RuntimeError):
            run(source)
    else:
        from torch.utils.data import Dataset, DataLoader

        tensors = loader().dataset

        class Half(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                if index == 2:
                    raise RuntimeError("explicit incomplete dataset")
                return tensors[index]

        with pytest.raises(RuntimeError):
            run(source, DataLoader(Half(), batch_size=2))
    assert set(_RESULTS) == before


@pytest.mark.parametrize(
    "field", ["expected_checkpoint_sha256", "expected_model_state_sha256"]
)
def test_wrong_expected_identity_rejected_on_opt_in(tmp_path, field):
    source = trusted(tmp_path)
    with pytest.raises(ValueError):
        run(source, **{field: "0" * 64})


def test_verified_runtime_does_not_override_incompatible_streams(tmp_path):
    a, b = trusted(tmp_path, "a"), trusted(tmp_path, "b")
    left, right = run(a), run(b, loader(shift=1))
    compared = compare_graph_evaluations(left, right).to_dict()
    assert all(
        v["inference_binding"] == "verified" for v in compared["verification"].values()
    )
    assert compared["status"] == "INCOMPATIBLE"
    assert all(row["difference"] is None for row in compared["metrics"])


def test_another_execution_statement_is_rejected_even_when_envelope_rehashed(tmp_path):
    from hyso.analysis._codec import digest

    source = trusted(tmp_path)
    left, right = run(source), run(source)
    data = left.to_dict()
    data["receiver_binding"] = right.to_dict()["receiver_binding"]
    data["content_sha256"] = digest(
        {
            k: v
            for k, v in data.items()
            if k not in ("content_sha256", "verification", "relation")
        }
    )
    with pytest.raises(ValueError, match="statement/evaluation"):
        GraphEvaluationResult(data)


def test_copy_of_native_evaluation_cannot_supply_authority(tmp_path):
    from hyso._graph_receiver import receiver_statement, receiver_verification

    source = trusted(tmp_path)
    result = evaluate_classification(
        source,
        loader(),
        class_names=["a", "b"],
        split="test",
        capture_evidence=True,
        capture_receiver_runtime=True,
    )
    statement = receiver_statement(result)
    assert receiver_verification(result, statement) == "verified"
    from dataclasses import FrozenInstanceError
    from hyso.evaluation import EvaluationResult

    try:
        cloned = copy.deepcopy(result)
    except FrozenInstanceError:
        # Frozen slotted legacy EvaluationResult already rejects deepcopy.
        # Reconstructing identical caller bytes must also have no authority.
        cloned = EvaluationResult(result.to_dict())
    assert receiver_verification(cloned, statement) == "unverified"


def test_new_process_json_does_not_recreate_receiver_authority(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    import hyso

    source = trusted(tmp_path)
    result = run(source)
    path = tmp_path / "association.json"
    result.to_json(path)
    code = (
        "import sys,json;sys.path.insert(0,sys.argv[2]);"
        "from hyso.analysis.graph_evidence import GraphEvaluationResult;"
        "print(json.dumps(GraphEvaluationResult.from_json(sys.argv[1]).to_dict()['verification']))"
    )
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            code,
            str(path),
            str(Path(hyso.__file__).resolve().parent.parent),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1"),
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)["receiver_runtime"] == "unverified"


def test_source_loss_and_result_lifetime_do_not_leave_authority(tmp_path, monkeypatch):
    import gc
    import weakref

    source = trusted(tmp_path)
    seen = observe_receiver(monkeypatch)
    result = run(source)
    source_ref = weakref.ref(source)
    del source
    gc.collect()
    assert source_ref() is None
    assert result.to_dict()["relation"]["status"] != "verified"
    del result
    gc.collect()
    assert seen[0]() is None


@pytest.mark.parametrize(
    "kind", ["source_alias", "internal_storage_alias", "partial_storage_overlap"]
)
def test_receiver_storage_isolation_is_independently_checked(
    tmp_path, monkeypatch, kind
):
    from hyso._graph_receiver import _ReceiverCapture

    source = trusted(tmp_path)
    # Equal values make storage aliasing the only injected mismatch.
    with torch.no_grad():
        source.graph.nodes["node_left"].weight.fill_(1)
        source.graph.nodes["node_head"].linear.weight.fill_(1)
    path = tmp_path / "equal.pth"
    save_training_checkpoint(path, source, graph_contract=True, include_rng=False)
    load_training_checkpoint(path, source, capture_evidence=True)
    actual = _ReceiverCapture.start

    def alias(capture, receiver):
        left = receiver.graph.nodes["node_left"]
        if kind == "source_alias":
            left.weight = torch.nn.Parameter(
                source.graph.nodes["node_left"].weight.detach()
            )
        elif kind == "internal_storage_alias":
            left.weight = torch.nn.Parameter(
                receiver.graph.nodes["node_head"].linear.weight.view_as(left.weight)
            )
        else:
            import numpy as np

            allocation = np.ones(5, dtype=np.float32)
            receiver.graph.nodes["node_head"].linear.weight = torch.nn.Parameter(
                torch.from_numpy(allocation[:4]).reshape(2, 2)
            )
            left.weight = torch.nn.Parameter(
                torch.from_numpy(allocation[1:]).reshape_as(left.weight)
            )
        return actual(capture, receiver)

    monkeypatch.setattr(_ReceiverCapture, "start", alias)
    with pytest.raises(ValueError, match="storage"):
        run(source)


def test_class_level_behavior_change_with_same_weights_is_rejected(
    tmp_path, monkeypatch
):
    from hyso.models.graph_ops import Fusion
    from hyso._evaluation_evidence import model_hash

    source = trusted(tmp_path)
    before = model_hash(source)
    monkeypatch.setattr(Fusion, "name", "multiply")
    assert model_hash(source) == before
    with pytest.raises(ValueError, match="callable changed"):
        run(source)


def test_source_mutation_during_dataset_iteration_is_rejected(tmp_path):
    from torch.utils.data import Dataset, DataLoader

    source = trusted(tmp_path)
    tensors = loader().dataset

    class Changing(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            if index == 0:
                mutate(source, "parameter")
            return tensors[index]

    with pytest.raises(ValueError):
        run(source, DataLoader(Changing(), batch_size=2))


@pytest.mark.parametrize(
    "kind", ["dtype", "geometry", "cuda_option", "tensor_subclass"]
)
def test_unsupported_input_cannot_publish_runtime_binding(tmp_path, kind):
    from torch.utils.data import DataLoader, TensorDataset

    source = trusted(tmp_path)
    if kind == "cuda_option":
        with pytest.raises(ValueError):
            evaluate_classification(
                source,
                loader(),
                class_names=["a", "b"],
                split="test",
                device="cuda",
                capture_evidence=True,
                capture_receiver_runtime=True,
            )
        return
    data = loader(dtype=torch.float64 if kind == "dtype" else torch.float32)
    if kind == "geometry":
        data = DataLoader(
            TensorDataset(torch.ones(4, 1, 5, 5), torch.tensor([0, 1, 0, 1])),
            batch_size=2,
        )
    elif kind == "tensor_subclass":

        class OutsideTensor(torch.Tensor):
            pass

        tensors = data.dataset.tensors

        def collate(batch):
            x = torch.stack([item[0] for item in batch]).as_subclass(OutsideTensor)
            y = torch.tensor([int(item[1]) for item in batch])
            return x, y

        data = DataLoader(TensorDataset(*tensors), batch_size=2, collate_fn=collate)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        run(source, data)


@pytest.mark.parametrize(
    "kind", ["known_constructor_swap", "unknown_constructor_subclass"]
)
def test_changed_constructor_table_cannot_redefine_the_supported_profile(
    tmp_path, monkeypatch, kind
):
    from hyso.models import graph_ops

    if kind == "known_constructor_swap":
        replacement = graph_ops.Multiply
    else:

        class Outside(graph_ops.Fusion):
            def forward(self, *inputs):
                return inputs[0] - inputs[1]

        replacement = Outside
    monkeypatch.setitem(graph_ops.CONSTRUCTORS, "add", replacement)
    # The old v2 validator reconstructs its reference from that table. A native
    # source receipt alone must not redefine the new receiver's installed profile.
    source = trusted(tmp_path)
    assert graph_checkpoint_evidence(source)["status"] == "native"
    with pytest.raises(ValueError):
        run(source)
