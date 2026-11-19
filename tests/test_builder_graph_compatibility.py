"""Bounded stage 13.5.12 architecture audit, no hardware/quality acceptance."""

import copy
import hashlib
import inspect
import json
import itertools
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from hyso import ModelBuilder, GraphModelBuilder, ModelRegistry, ModelConfigError
from hyso.checkpoint import (
    save_training_checkpoint,
    load_training_checkpoint,
    TrainingCheckpointError,
)
from hyso.models.output import GraphTensorOutput
from hyso.architecture_comparison import compare_architectures
from hyso.models._profile import DEVELOPMENT

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/builder-graph-compatibility-v1.json").read_text(
        encoding="utf8"
    )
)
SAMPLES = FIXTURE["component_samples"]
AVAILABLE = (
    {row["name"] for row in FIXTURE["registry"]}
    if DEVELOPMENT
    else set(FIXTURE["commercial_names"])
)
PAIRWISE = [
    (c, a, n, (c + a + n) % 3) for c in range(3) for a in range(3) for n in range(2)
]


def tensor(shape=(2, 4, 8, 8), dtype=torch.float32):
    return torch.randn(
        shape,
        generator=torch.Generator().manual_seed(13512),
        dtype=dtype,
        requires_grad=True,
    )


def exercise(model, fresh, inputs, tmp_path, *, select=lambda value: value):
    model.train()
    parameters = dict(model.named_parameters())
    before = {name: value.detach().clone() for name, value in parameters.items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) if parameters else None
    output = select(model(inputs))
    assert type(output) is torch.Tensor and torch.isfinite(output).all()
    weights = torch.linspace(0.5, 1.5, output.numel(), dtype=output.dtype).reshape_as(
        output
    )
    loss = ((output - weights) ** 2).mean()
    assert torch.isfinite(loss)
    loss.backward()
    input_values = inputs.values() if isinstance(inputs, dict) else [inputs]
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all()
        for value in input_values
    )
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in parameters.values()
        if p.requires_grad
    )
    if optimizer:
        optimizer.step()
        assert any(
            not torch.equal(before[name], value) for name, value in parameters.items()
        )
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    path = tmp_path / "state.pth"
    save_training_checkpoint(path, model, include_rng=False)
    other = fresh()
    load_training_checkpoint(
        path, other, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert set(other.state_dict()) == set(model.state_dict())
    assert all(
        torch.equal(other.state_dict()[k], v) for k, v in model.state_dict().items()
    )
    model.eval()
    other.eval()
    with torch.no_grad():
        assert torch.equal(select(model(inputs)), select(other(inputs)))
    return dict(
        shape=list(output.shape),
        parameters=sum(p.numel() for p in parameters.values()),
        trainable=sum(p.numel() for p in parameters.values() if p.requires_grad),
        input_gradient=True,
        finite_gradients=True,
        optimizer_updated=bool(parameters),
        strict_checkpoint=True,
    )


def component_builder(name, engine):
    params = SAMPLES[name]
    if engine == "sequential":
        if name.endswith("_head"):
            return ModelBuilder.sequential(
                (4, 8, 8),
                head=name[:-5],
                **{k: v for k, v in params.items() if k != "in_features"},
            )
        if name == "flatten":
            return ModelBuilder.sequential(
                (4, 8, 8), num_classes=3, pooling="none"
            ).add(name, **params)
        return ModelBuilder((4, 8, 8), 3).add(name, **params)
    return (
        GraphModelBuilder({"image": [4, 8, 8]})
        .add("feature", name, ["image"], **params)
        .outputs(features="feature")
    )


def test_a_complete_catalog_is_backed_by_actual_constructors():
    registry = ModelRegistry()
    actual = [spec.to_dict() for spec in registry.list()]
    expected = [row for row in FIXTURE['registry'] if row['name'] in AVAILABLE]
    # UX-03/05 expanded public docstrings before UX-06; keep the golden fixture.
    # Compare every structural field, including the nested parameter descriptions.
    def contract(rows):
        return [{key:value for key,value in row.items() if key!='description'} for row in rows]
    assert contract(actual) == contract(expected)
    assert all(isinstance(row['description'],str) and row['description'] for row in actual)
    assert len(registry.list()) == (85 if DEVELOPMENT else 57)
    assert set(SAMPLES) & AVAILABLE == {
        s.name for s in registry.list() if s.category != "model"
    }
    for spec in registry.list():
        signature = inspect.signature(spec.constructor)
        assert callable(spec.constructor) and spec.constructor.__module__
        assert all(
            p.name in signature.parameters
            or any(v.kind == v.VAR_KEYWORD for v in signature.parameters.values())
            for p in spec.parameters
        )


@pytest.mark.parametrize("engine", ["registry", "sequential", "graph"])
@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_b_c_e_component_composition_state_and_optimizer(
    name, engine, tmp_path, record_property
):
    registry = ModelRegistry()
    if name not in AVAILABLE:
        with pytest.raises(ModelConfigError):
            if engine == "registry":
                registry.create(name, **SAMPLES[name])
            else:
                component_builder(name, engine)
        record_property(
            "acceptance",
            json.dumps(
                dict(
                    component=name,
                    engine=engine,
                    status="UNSUPPORTED",
                    reason="development_component_excluded_from_commercial_registry",
                )
            ),
        )
        return
    if engine == "registry":

        def fresh():
            return registry.create(name, **SAMPLES[name])

        expected = component_builder(name, "graph").preview()["nodes"]["feature"][
            "output_shape"
        ][1:]
    else:
        builder = component_builder(name, engine)
        config = builder.to_config()
        parsed = type(builder).from_config(json.loads(json.dumps(config)))
        assert parsed.to_config() == config
        file = tmp_path / "architecture.json"
        builder.to_json(file)
        assert type(builder).from_json(file).to_config() == config
        if engine == "graph":

            def fresh():
                return GraphTensorOutput(parsed.build(), "features")

            expected = builder.preview()["nodes"]["feature"]["output_shape"][1:]
        else:
            fresh = parsed.build
            expected = ModelBuilder.preflight(config)["output_shape"][1:]
    model = fresh()
    result = exercise(model, fresh, tensor(), tmp_path)
    assert result["shape"] == [2, *expected]
    record_property(
        "acceptance",
        json.dumps(dict(component=name, engine=engine, status="SUPPORTED", **result)),
    )


def topology(case):
    b = GraphModelBuilder({"image": [4, 8, 8]})
    if case == "sequential":
        b.add("stem", "conv2d", ["image"], in_channels=4, out_channels=4)
        last, channels = "stem", 4
    elif case == "residual":
        b.add("stem", "conv2d", ["image"], in_channels=4, out_channels=4).add(
            "merge", "add", ["stem", "image"]
        )
        last, channels = "merge", 4
    else:
        b.add("stem", "conv_norm_act", ["image"], in_channels=4, out_channels=4)
        b.add("left", "conv2d", ["stem"], in_channels=4, out_channels=4)
        b.add("right", "pointwise_conv2d", ["stem"], in_channels=4, out_channels=4)
        if case == "unequal_depth":
            b.add("deep", "conv2d", ["left"], in_channels=4, out_channels=4)
            refs = ["deep", "right"]
        elif case == "multi_conv":
            b.add("third", "conv2d", ["stem"], in_channels=4, out_channels=4, groups=2)
            refs = ["left", "right", "third"]
        else:
            refs = ["left", "right"]
        name = (
            "concat"
            if case in ("concat", "multi_conv")
            else case
            if case in ("multiply", "weighted_sum")
            else "add"
        )
        params = (
            {"axis": 1}
            if name == "concat"
            else {"weights": [0.25, 0.75], "learnable": True}
            if name == "weighted_sum"
            else {}
        )
        b.add("merge", name, refs, **params)
        last, channels = "merge", 4 * len(refs) if name == "concat" else 4
        if case == "nested":
            b.add("deep", "conv2d", ["merge"], in_channels=4, out_channels=4)
            b.add("activated", "relu", ["merge"]).add(
                "joined", "concat", ["deep", "activated"], axis=1
            )
            last, channels = "joined", 8
    return b.add(
        "head", "classification_head", [last], in_features=channels, num_classes=3
    ).outputs(logits="head")


@pytest.mark.parametrize(
    "case",
    [
        "sequential",
        "branching",
        "residual",
        "concat",
        "multi_conv",
        "unequal_depth",
        "nested",
        "multiply",
        "weighted_sum",
    ],
)
def test_c_graph_topologies_register_every_parameter_and_restore(
    case, tmp_path, record_property
):
    builder = topology(case)
    assert builder.preview()["valid"]
    config = builder.to_config()
    normalized = GraphModelBuilder.from_config(json.loads(json.dumps(config)))

    def fresh():
        return GraphTensorOutput(normalized.build(), "logits", task="multiclass")

    result = exercise(fresh(), fresh, tensor(), tmp_path)
    assert result["shape"] == [2, 3]
    record_property(
        "acceptance", json.dumps(dict(topology=case, status="SUPPORTED", **result))
    )


@pytest.mark.parametrize(
    "channels,spatial,classes",
    [(1, [5, 7], 1), (3, [16, 20], 2), (7, [9, 11], 5), (4, [32, 24], 3)],
)
def test_b_variable_geometry_and_head_width(
    channels, spatial, classes, tmp_path, record_property
):
    builder = (
        ModelBuilder((channels, *spatial), classes)
        .add("conv2d", in_channels=channels, out_channels=8)
        .add("relu")
        .add("max_pool2d", kernel_size=2, stride=2)
        .add("conv_norm_act", in_channels=8, out_channels=4, norm="group")
    )
    result = exercise(
        builder.build(), builder.build, tensor((2, channels, *spatial)), tmp_path
    )
    assert result["shape"] == [2, classes]
    record_property(
        "acceptance",
        json.dumps(
            dict(
                case="geometry",
                channels=channels,
                spatial=spatial,
                classes=classes,
                **result,
            )
        ),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "kernel,stride,padding,dilation,groups",
    [
        (3, 1, 1, 1, 1),
        (3, 2, 2, 2, 2),
        (2, 2, 0, 1, 2),
        ([3, 5], [2, 3], [2, 1], [2, 1], 2),
    ],
)
def test_d_independent_convolution_geometry(
    dtype, kernel, stride, padding, dilation, groups
):
    def pair(value):
        return [value, value] if isinstance(value, int) else value

    shape = [13, 17]
    expected = [
        (v + 2 * p - d * (k - 1) - 1) // s + 1
        for v, p, d, k, s in zip(
            shape, pair(padding), pair(dilation), pair(kernel), pair(stride)
        )
    ]
    b = ModelBuilder((4, *shape), 3).add(
        "conv2d",
        in_channels=4,
        out_channels=6,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    report = ModelBuilder.preflight(b.to_config())
    actual = b.build().to(dtype).features(tensor((2, 4, *shape), dtype=dtype))
    assert list(actual.shape) == [2, 6, *expected] == [2, *report["feature_shape"][1:]]


@pytest.mark.parametrize("name", ["max_pool2d", "avg_pool2d"])
@pytest.mark.parametrize("ceil_mode", [False, True])
def test_d_independent_pooling_geometry(name, ceil_mode):
    b = ModelBuilder((4, 7, 9), 3).add(
        name, kernel_size=2, stride=2, ceil_mode=ceil_mode
    )
    expected = [4, 5] if ceil_mode else [3, 4]
    assert list(b.build().features(tensor((2, 4, 7, 9))).shape) == [2, 4, *expected]
    assert ModelBuilder.preflight(b.to_config())["feature_shape"] == ["N", 4, *expected]


@pytest.mark.parametrize("batch", [1, 2, 5])
def test_d_graph_batch_is_dynamic_but_spatial_contract_is_fixed(batch):
    graph = topology("branching").build()
    assert graph(tensor((batch, 4, 8, 8)))["logits"].shape == (batch, 3)
    with pytest.raises(ModelConfigError):
        graph(tensor((batch, 4, 9, 8)))


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "cycle",
        "self_loop",
        "disconnected",
        "no_output",
        "add_shape",
        "concat_shape",
        "unary_fanin",
        "head_features",
    ],
)
def test_i_invalid_graph_is_rejected_before_forward(bad, record_property):
    b = GraphModelBuilder({"image": [4, 8, 8]})
    if bad == "missing":
        b.add("a", "identity", ["missing"]).outputs(out="a")
    elif bad == "cycle":
        b.add("a", "identity", ["b"]).add("b", "identity", ["a"]).outputs(out="a")
    elif bad == "self_loop":
        b.add("a", "identity", ["a"]).outputs(out="a")
    elif bad == "no_output":
        b.add("a", "identity", ["image"])
    elif bad == "disconnected":
        b.add("a", "identity", ["image"]).add("unused", "relu", ["image"]).outputs(
            out="a"
        )
    elif bad == "head_features":
        b.add(
            "head", "classification_head", ["image"], in_features=9, num_classes=3
        ).outputs(out="head")
    elif bad == "unary_fanin":
        b.add("a", "relu", ["image", "image"]).outputs(out="a")
    else:
        b.add("small", "conv2d", ["image"], in_channels=4, out_channels=1, stride=2)
        b.add(
            "a", "add" if bad == "add_shape" else "concat", ["image", "small"]
        ).outputs(out="a")
    report = b.preview()
    assert not report["valid"] and report["diagnostics"]
    diagnostic = report["diagnostics"][0]
    assert diagnostic["message_en"] and diagnostic["message_tr"] and diagnostic["code"]
    with pytest.raises(ModelConfigError):
        b.build()
    record_property(
        "acceptance",
        json.dumps(
            dict(
                case=bad,
                status="INVALID_CONFIGURATION",
                phase="preflight/build",
                diagnostics=report["diagnostics"],
            )
        ),
    )


@pytest.mark.parametrize(
    "name,params",
    [
        ("conv2d", dict(in_channels=0, out_channels=4)),
        ("conv2d", dict(in_channels=4, out_channels=4, kernel_size=0)),
        ("conv2d", dict(in_channels=4, out_channels=6, groups=4)),
        ("conv2d", dict(in_channels=4, out_channels=4, dilation=0)),
        ("conv2d", dict(in_channels=4, out_channels=4, bias=1)),
        ("group_norm", dict(num_groups=3, num_channels=4)),
        ("prelu", dict(num_parameters=3)),
        ("pixel_shuffle", dict(upscale_factor=3)),
        ("resize", dict(scale_factor=0)),
        ("classification_head", dict(in_features=4, num_classes=0)),
        ("unknown", {}),
    ],
)
def test_i_scalar_and_composition_errors_preserve_builder(name, params):
    b = ModelBuilder((4, 8, 8), 3)
    before = b.to_config()
    with pytest.raises(ModelConfigError):
        b.add(name, **params)
    assert b.to_config() == before


def test_i_duplicate_identifiers_and_batch_concat_axis_are_rejected():
    b = GraphModelBuilder({"image": [4, 8, 8]}).add("a", "identity", ["image"])
    with pytest.raises(ModelConfigError):
        b.add("a", "relu", ["image"])
    with pytest.raises(ModelConfigError):
        b.add("joined", "concat", ["a", "image"], axis=0)


def test_c_multi_input_raw_graph_and_adapter_boundary(tmp_path):
    b = (
        GraphModelBuilder({"left": [4, 8, 8], "right": [4, 8, 8]})
        .add("merge", "add", ["left", "right"])
        .add("head", "classification_head", ["merge"], in_features=4, num_classes=3)
        .outputs(logits="head")
    )
    raw = b.build()
    with pytest.raises(ValueError):
        GraphTensorOutput(raw, "logits")
    exercise(
        raw,
        b.build,
        {"left": tensor(), "right": tensor()},
        tmp_path,
        select=lambda out: out["logits"],
    )


def test_c_unused_named_output_is_valid_but_has_no_selected_loss_gradient():
    b = (
        topology("branching")
        .add("side", "conv2d", ["image"], in_channels=4, out_channels=4)
        .outputs(logits="head", side="side")
    )
    assert b.preview()["valid"]
    m = GraphTensorOutput(b.build(), "logits")
    m(tensor()).square().mean().backward()
    assert all(p.grad is None for p in m.graph.nodes["node_side"].parameters())
    assert all(p.grad is not None for p in m.graph.nodes["node_head"].parameters())


def test_f_valid_general_graph_does_not_gain_unsupported_v2_trust(tmp_path):
    from hyso.analysis.graph_evidence import evaluate_graph_classification

    b = (
        GraphModelBuilder({"image": [4, 8, 8]})
        .add("depth", "depthwise_conv2d", ["image"], in_channels=4)
        .add("head", "classification_head", ["depth"], in_features=4, num_classes=3)
        .outputs(logits="head")
    )
    model = GraphTensorOutput(b.build(), "logits")
    with pytest.raises(TrainingCheckpointError):
        save_training_checkpoint(tmp_path / "v2.pth", model, graph_contract=True)
    data = DataLoader(
        TensorDataset(tensor().detach(), torch.tensor([0, 1])), batch_size=2
    )
    with pytest.raises(ValueError):
        evaluate_graph_classification(
            model,
            data,
            class_names=["a", "b", "c"],
            split="test",
            capture_receiver_runtime=True,
        )
    result = evaluate_graph_classification(
        model, data, class_names=["a", "b", "c"], split="test"
    ).to_dict()
    assert result["verification"]["graph_v2"] == "missing"
    assert (
        result["evaluation"]["producer_evidence"]["checkpoint"]["status"] == "missing"
    )


def test_g_metadata_identity_does_not_identify_weights():
    from hyso import capture_model_structure, compare_model_structures

    b = ModelBuilder((4, 8, 8), 3).add("conv2d", in_channels=4, out_channels=4)
    a, c = b.build(), b.build()
    with torch.no_grad():
        next(c.parameters()).add_(1)
    assert compare_architectures([a.to_config(), c.to_config()])[
        "equivalent_groups"
    ] == [[0, 1]]
    assert (
        capture_model_structure(a).content_sha256
        == capture_model_structure(c).content_sha256
    )
    compared = compare_model_structures(
        capture_model_structure(a), capture_model_structure(c)
    ).to_dict()
    assert compared["format"] == "hyso.model_structure_comparison"


@pytest.mark.parametrize(
    "change", ["activation", "topology", "channels", "head", "extra_conv"]
)
def test_g_architecture_comparison_observes_declared_changes(change):
    a = topology("branching").to_config()
    b = copy.deepcopy(a)
    if change == "activation":
        next(row for row in b["nodes"] if row["id"] == "stem")["params"][
            "activation"
        ] = "relu"
    elif change == "topology":
        next(row for row in b["nodes"] if row["id"] == "merge")["component"] = (
            "multiply"
        )
    elif change == "head":
        next(row for row in b["nodes"] if row["id"] == "head")["params"][
            "num_classes"
        ] = 5
    elif change == "channels":
        b = (
            ModelBuilder((4, 8, 8), 3)
            .add("conv2d", in_channels=4, out_channels=8)
            .to_config()
        )
    else:
        b = (
            ModelBuilder((4, 8, 8), 3)
            .add("conv2d", in_channels=4, out_channels=4)
            .add("conv2d", in_channels=4, out_channels=4)
            .to_config()
        )
    result = compare_architectures([a, b], exact_parameters=True)
    assert all(column["valid"] for column in result["columns"])
    assert result["equivalent_groups"] == []
    assert all(column["parameter_count"] is not None for column in result["columns"])
    assert not any("winner" in key for key in result)


@pytest.mark.parametrize(
    "name", [row["name"] for row in FIXTURE["registry"] if row["category"] == "model"]
)
def test_a_named_families_are_profile_gated_and_not_graph_nodes(name):
    if name == "modular_cnn":
        builder = ModelBuilder.modular(
            input_channels=4,
            num_classes=3,
            widths=[4, 8],
            repeats=[1, 1],
            strides=[1, 2],
        )
    elif name in AVAILABLE:
        builder = ModelBuilder.named(name, input_channels=1, num_classes=3)
    else:
        with pytest.raises(ModelConfigError):
            ModelBuilder.named(name, input_channels=1, num_classes=3)
        return
    assert ModelBuilder.preflight(builder.to_config())["valid"]
    params = builder.to_config()["params"]
    graph = (
        GraphModelBuilder({"image": [4, 8, 8]})
        .add("family", name, ["image"], **params)
        .outputs(out="family")
    )
    assert not graph.preview()["valid"]
    with pytest.raises(ModelConfigError):
        graph.build()


def test_f_binary_head_integrates_with_existing_training_and_evaluation():
    from hyso import TrainingConfig, evaluate_classification

    builder = (
        ModelBuilder((1, 4, 4), 1)
        .add("conv2d", in_channels=1, out_channels=2)
        .add("relu")
    )
    model = builder.build()
    data = DataLoader(
        TensorDataset(
            torch.arange(64, dtype=torch.float32).reshape(4, 1, 4, 4) / 64,
            torch.tensor([0.0, 1.0, 0.0, 1.0]),
        ),
        batch_size=2,
    )
    result = (
        TrainingConfig(
            epochs=1,
            batch_size=2,
            device="cpu",
            precision="fp32",
            task="binary",
            shuffle=False,
            validation=False,
            optimizer="sgd",
        )
        .prepare(model, data)
        .fit()
    )
    assert result.to_dict()["applied_updates"] == 2
    evaluation = evaluate_classification(
        model,
        data,
        class_names=["negative", "positive"],
        split="synthetic",
        task="binary",
    )
    assert evaluation.to_dict()["metrics"]["sample_count"] == 4


def test_b_pairwise_schedule_covers_every_factor_pair_but_not_the_whole_universe():
    values = [range(3), range(3), range(2), range(3)]
    assert len(PAIRWISE) == 18 and 3 * 3 * 2 * 3 - len(PAIRWISE) == 36
    for left, right in itertools.combinations(range(4), 2):
        assert {(row[left], row[right]) for row in PAIRWISE} == set(
            itertools.product(values[left], values[right])
        )


@pytest.mark.parametrize("conv,activation,norm,pool", PAIRWISE)
def test_b_e_risk_pairwise_cnn_combinations(
    conv, activation, norm, pool, tmp_path, record_property
):
    b = ModelBuilder((4, 8, 8), 3)
    conv_name = ["conv2d", "conv2d", "depthwise_conv2d"][conv]
    params = dict(in_channels=4)
    if conv < 2:
        params.update(out_channels=4, groups=2 if conv == 1 else 1)
    b.add(conv_name, **params)
    if norm == 0:
        b.add("batch_norm", num_features=4)
    else:
        b.add("group_norm", num_groups=2, num_channels=4)
    b.add(["relu", "silu", "gelu"][activation])
    b.add(
        ["max_pool2d", "avg_pool2d", "adaptive_avg_pool2d"][pool],
        **({"kernel_size": 2, "stride": 2} if pool < 2 else {"output_size": [4, 4]}),
    )
    result = exercise(b.build(), b.build, tensor(), tmp_path)
    assert result["shape"] == [2, 3]
    record_property(
        "acceptance",
        json.dumps(
            dict(
                case="pairwise",
                factors=[conv, activation, norm, pool],
                status="SUPPORTED",
                **result,
            )
        ),
    )


@pytest.mark.parametrize("fusion", ["add", "multiply", "concat", "weighted_sum"])
def test_i_graph_fusions_are_not_linear_registry_components(fusion):
    with pytest.raises(ModelConfigError):
        ModelRegistry().get(fusion)
    with pytest.raises(ModelConfigError):
        ModelBuilder((4, 8, 8), 3).add(fusion)


def test_h_config_is_detached_and_rebuilding_requires_explicit_seed_for_equal_weights():
    b = ModelBuilder((4, 8, 8), 3).add("conv2d", in_channels=4, out_channels=4)
    config = b.to_config()
    config["blocks"][0]["params"]["out_channels"] = 99
    assert b.to_config()["blocks"][0]["params"]["out_channels"] == 4
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(13512)
        a = b.build()
        torch.random.default_generator.manual_seed(13512)
        c = ModelBuilder.from_config(b.to_config()).build()
        assert all(
            torch.equal(value, c.state_dict()[name])
            for name, value in a.state_dict().items()
        )
        other = b.build()
        assert any(
            not torch.equal(value, other.state_dict()[name])
            for name, value in a.state_dict().items()
        )


@pytest.mark.parametrize("engine", ["sequential", "graph"])
def test_d_symbolic_batch_normalization_requirement_is_runtime_conditional(engine):
    if engine == "sequential":
        b = ModelBuilder((4, 1, 1), 3).add("batch_norm", num_features=4)
        report = ModelBuilder.preflight(b.to_config())
        m = b.build()
    else:
        b = (
            GraphModelBuilder({"image": [4, 1, 1]})
            .add("norm", "batch_norm", ["image"], num_features=4)
            .add("head", "classification_head", ["norm"], in_features=4, num_classes=3)
            .outputs(logits="head")
        )
        report = b.preview()
        m = GraphTensorOutput(b.build(), "logits")
    assert report["valid"] and report["warnings"]
    with pytest.raises(ModelConfigError):
        m(torch.ones(1, 4, 1, 1))
    m.eval()
    assert m(torch.ones(1, 4, 1, 1)).shape == (1, 3)


def test_h_output_adapter_task_guards_are_explicit():
    graph = topology("branching").build()
    with pytest.raises(ValueError):
        GraphTensorOutput(graph, "missing")
    with pytest.raises(ValueError):
        GraphTensorOutput(graph, "logits", task="binary")
