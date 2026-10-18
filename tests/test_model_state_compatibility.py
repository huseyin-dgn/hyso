"""P0.1-A structural references, not historical checkpoint compatibility."""

import pytest
import torch

from hyso.models import EBO


def describe_state(model):
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    described = []
    for key, tensor in model.state_dict().items():
        if key in parameters:
            kind = "parameter"
        elif key in buffers:
            kind = "buffer"
        else:
            pytest.fail(f"Unclassified state entry: {key}")
        described.append({
            "key": key,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "kind": kind,
            "requires_grad": parameters[key].requires_grad if kind == "parameter" else False,
        })
    return described


def inference_target(model):
    # EBO has no wrapper forward (R01); use the explicitly documented child.
    return model.model if isinstance(model, EBO) else model


def test_state_schema_matches_frozen_manifest(model_case, make_model):
    model = make_model(model_case)
    assert describe_state(model) == model_case["state_dict"]


def test_strict_same_configuration_load_preserves_eval_output(
    model_case, make_model, cpu_input,
):
    source = make_model(model_case).eval()
    target = make_model(model_case).eval()
    # The RNG is not reset between constructors: target is independently
    # initialized. Loading must restore real floating state, not just a schema.
    source_state = source.state_dict()
    target_before = target.state_dict()
    assert any(
        not torch.equal(source_state[key], target_before[key])
        for key in source_state if source_state[key].is_floating_point()
    )
    result = target.load_state_dict(source_state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    for key, tensor in source_state.items():
        torch.testing.assert_close(target.state_dict()[key], tensor, rtol=0, atol=0)
    with torch.no_grad():
        expected = inference_target(source)(cpu_input)
        actual = inference_target(target)(cpu_input)
    assert expected.shape == actual.shape == (2, 2)
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "wrapper_id,child_id",
    [("EBO_auto", "ELO_auto"), ("EBO_batch_depthwise", "ELO_batch_depthwise")],
)
def test_ebo_wrapper_prefix_and_child_schema(
    wrapper_id, child_id, manifest_cases, make_model,
):
    by_id = {case["id"]: case for case in manifest_cases}
    wrapper = make_model(by_id[wrapper_id])
    wrapper_schema = describe_state(wrapper)
    assert wrapper_schema
    assert all(entry["key"].startswith("model.") for entry in wrapper_schema)
    assert wrapper_schema == by_id[wrapper_id]["state_dict"]
    child_schema = describe_state(wrapper.model)
    assert all(not entry["key"].startswith("model.") for entry in child_schema)
    assert child_schema == by_id[child_id]["state_dict"]
    assert [dict(entry, key=entry["key"][6:]) for entry in wrapper_schema] == child_schema


@pytest.mark.parametrize("wrapper_id", ["EBO_auto", "EBO_batch_depthwise"])
@pytest.mark.parametrize("direction", ["wrapper-to-child", "child-to-wrapper"])
def test_ebo_wrong_strict_load_target_is_rejected(
    wrapper_id, direction, manifest_cases, make_model,
):
    case = next(case for case in manifest_cases if case["id"] == wrapper_id)
    wrapper = make_model(case)
    if direction == "wrapper-to-child":
        source, target = wrapper, wrapper.model
    else:
        source, target = wrapper.model, wrapper
    with pytest.raises(RuntimeError) as error:
        target.load_state_dict(source.state_dict(), strict=True)
    message = str(error.value)
    assert "Missing key(s) in state_dict" in message
    assert "Unexpected key(s) in state_dict" in message
