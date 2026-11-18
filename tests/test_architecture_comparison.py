"""AUDIT-01: preserve Core diagnostics before reading invalid architecture fields."""

import copy
import inspect

import pytest
import torch

from hyso import GraphModelBuilder, ModelBuilder
from hyso.architecture_comparison import compare_architectures
from hyso.model_checks import config_digest, preflight_model


def graph():
    return (
        GraphModelBuilder({"image": [1, 2, 2]})
        .add("head", "classification_head", ["image"], in_features=1, num_classes=2)
        .outputs(logits="head")
        .to_config()
    )


def invalid_config(case):
    if case == "schema1_head_string":
        config = ModelBuilder((1, 2, 2), 2).to_config()
        config["head"] = "invalid"
    elif case.startswith("schema2_"):
        config = ModelBuilder.sequential((1, 2, 2), num_classes=2).to_config()
        if case == "schema2_head_string":
            config["head"] = "invalid"
        elif case == "schema2_head_none":
            config["head"] = None
        elif case == "schema2_head_list":
            config["head"] = []
        elif case == "schema2_head_params_none":
            config["head"]["params"] = None
        elif case == "schema2_head_params_list":
            config["head"]["params"] = []
    else:
        config = graph()
        if case == "schema3_nodes_none":
            config["nodes"] = None
        elif case == "schema3_nodes_string":
            config["nodes"] = "invalid"
        elif case == "schema3_nodes_dict":
            config["nodes"] = {}
        elif case == "schema3_node_none":
            config["nodes"] = [None]
        elif case == "schema3_node_string":
            config["nodes"] = ["invalid"]
        elif case == "schema3_params_none":
            config["nodes"][0]["params"] = None
        elif case == "schema3_params_list":
            config["nodes"][0]["params"] = []
        elif case == "schema3_inputs_none":
            config["nodes"][0]["inputs"] = None
        elif case == "schema3_head_features_mismatch":
            config["nodes"][0]["params"]["in_features"] = 9
        else:
            raise AssertionError("Unknown fixture")
    return config


CASES = [
    "schema1_head_string",
    "schema2_head_string",
    "schema2_head_none",
    "schema2_head_list",
    "schema2_head_params_none",
    "schema2_head_params_list",
    "schema3_nodes_none",
    "schema3_nodes_string",
    "schema3_nodes_dict",
    "schema3_node_none",
    "schema3_node_string",
    "schema3_params_none",
    "schema3_params_list",
    "schema3_inputs_none",
    "schema3_head_features_mismatch",
]


def assert_invalid(column, report):
    assert column["valid"] is False
    # Entire original diagnostic dictionaries, including code, context and text.
    assert column["diagnostics"] == report["diagnostics"]
    assert column["diagnostics"]
    assert column["parameter_status"] == "UNAVAILABLE"
    for field in (
        "parameter_count",
        "parameter_bound",
        "parameter_delta",
        "activation_estimate",
        "node_count",
        "branch_count",
        "connections",
    ):
        assert column[field] is None, field
    assert column["mechanisms"] == column["categories"] == {}


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("exact", [False, True])
def test_invalid_containers_preserve_core_without_allocation(case, exact, monkeypatch):
    invalid = invalid_config(case)
    before = copy.deepcopy(invalid)
    report = preflight_model(invalid)
    assert report["valid"] is False and report["diagnostics"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid comparisons must not build/count models")

    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    monkeypatch.setattr(torch, "empty", forbidden)
    result = compare_architectures(
        [invalid, copy.deepcopy(invalid)], exact_parameters=exact
    )
    for column in result["columns"]:
        assert_invalid(column, report)
        assert column["config_sha256"] == config_digest(invalid)
    assert result["input_geometry_matches"] is False
    assert invalid == before


@pytest.mark.parametrize("exact", [False, True])
def test_mixed_models_preserve_each_diagnostic_and_valid_output(exact):
    valid = graph()
    second = copy.deepcopy(valid)
    second["nodes"][0]["params"]["num_classes"] = 3
    invalids = [
        invalid_config(case)
        for case in (
            "schema1_head_string",
            "schema3_nodes_none",
            "schema3_head_features_mismatch",
        )
    ]
    original = copy.deepcopy([valid, *invalids, second])
    reference = compare_architectures(
        [valid, second], names=["A", "B"], exact_parameters=exact
    )
    result = compare_architectures(
        [valid, *invalids, second],
        names=["A", "bad1", "bad2", "bad3", "B"],
        exact_parameters=exact,
    )
    assert result["columns"][0] == reference["columns"][0]
    assert result["columns"][-1] == reference["columns"][1]
    for column, invalid in zip(result["columns"][1:-1], invalids):
        assert_invalid(column, preflight_model(invalid))
    assert [valid, *invalids, second] == original


def test_invalid_baseline_does_not_invent_delta_for_valid_model():
    result = compare_architectures(
        [invalid_config("schema3_nodes_none"), graph()], exact_parameters=True
    )
    assert result["columns"][1]["valid"] is True
    assert result["columns"][1]["parameter_count"] == 4
    assert all(c["parameter_delta"] is None for c in result["columns"])


def test_valid_schema123_comparison_shape_and_counts_are_unchanged():
    first = ModelBuilder((1, 2, 2), 2).to_config()
    second = ModelBuilder.sequential((1, 2, 2), num_classes=2).to_config()
    result = compare_architectures([first, second, graph()], exact_parameters=True)
    assert result["format"] == "hyso.architecture_comparison"
    assert result["schema_version"] == 1
    assert [c["parameter_count"] for c in result["columns"]] == [4, 4, 4]
    assert [c["parameter_delta"] for c in result["columns"]] == [0, 0, 0]
    assert all(
        c["valid"] and c["parameter_status"] == "EXACT" and c["diagnostics"] == []
        for c in result["columns"]
    )
    assert list(inspect.signature(compare_architectures).parameters) == [
        "configs",
        "names",
        "exact_parameters",
    ]
    expected_keys = {
        "name",
        "config_sha256",
        "valid",
        "inputs",
        "outputs",
        "output_shape",
        "node_count",
        "connections",
        "branch_count",
        "mechanisms",
        "categories",
        "parameter_count",
        "parameter_bound",
        "parameter_status",
        "parameter_reason",
        "activation_estimate",
        "diagnostics",
        "parameter_delta",
    }
    assert all(set(c) == expected_keys for c in result["columns"])
