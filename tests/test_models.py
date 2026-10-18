"""Small CPU model smoke/gradient tests; no fit or training-loop regressions."""

import pytest
import torch
from torch import nn

from hyso.models import CEMO, CNNSENormalWithFit, EBO, ELO
from regression_guards import R01ForwardMissing


def tiny_cbam_config():
    return dict(
        input_channels=3, num_classes=2, conv_channels=[4, 8],
        residual_per_block=1, p_drop=0.0, att_reduction=2,
        norm_type="auto", use_depthwise=False,
    )


def tiny_cemo_config():
    return dict(
        input_channels=3, num_classes=2, conv_channels=[4, 8],
        residuals_per_stage=1, se_reduction=2, p_drop=0.0,
        dropout_fc=0.0, use_ema=False, ema_decay=0.999, device="cpu",
    )


@pytest.mark.parametrize("name", ["ELO", "CEMO", "EBO-child"])
def test_cpu_forward_backward(name, cpu_input):
    if name == "ELO":
        model = ELO(**tiny_cbam_config())
    elif name == "CEMO":
        model = CEMO(**tiny_cemo_config())
    else:
        wrapper = EBO(**tiny_cbam_config(), device="cpu")
        assert isinstance(wrapper, nn.Module)
        assert isinstance(wrapper.model, ELO)
        model = wrapper.model

    assert isinstance(model, nn.Module)
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert all(b.device.type == "cpu" for b in model.buffers())
    model.train()
    inputs = cpu_input.clone().requires_grad_(True)
    logits = model(inputs)
    assert logits.shape == (2, 2)
    assert logits.device.type == "cpu"
    assert torch.isfinite(logits).all()

    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 1], device="cpu"))
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g).item() > 0 for g in gradients)
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert torch.count_nonzero(inputs.grad).item() > 0


def test_cnns_constructor_smoke_cpu():
    model = CNNSENormalWithFit(**tiny_cemo_config())
    assert isinstance(model, CEMO)
    assert isinstance(model, nn.Module)
    assert model.num_classes == 2
    assert torch.device(model.device).type == "cpu"
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert all(b.device.type == "cpu" for b in model.buffers())
    assert model.optimizer is None
    assert model._callbacks == []


def test_ebo_direct_call_current_characterization(cpu_input):
    model = EBO(**tiny_cbam_config(), device="cpu").eval()
    with pytest.raises(NotImplementedError, match='missing the required "forward"'):
        model(cpu_input)


@pytest.mark.xfail(
    strict=True, reason="R01: EBO forward is missing", raises=R01ForwardMissing,
)
def test_ebo_direct_call_expected_contract(cpu_input):
    model = EBO(**tiny_cbam_config(), device="cpu").eval()
    with torch.no_grad():
        expected = model.model(cpu_input)
        assert expected.shape == (2, 2) and torch.isfinite(expected).all()
        try:
            logits = model(cpu_input)
        except NotImplementedError as error:
            # Only recognize the actual inherited missing-forward defect.
            if (type(model).forward is nn.Module.forward and
                    str(error) == 'Module [EBO] is missing the required "forward" function'):
                raise R01ForwardMissing(str(error)) from error
            raise
    assert logits.shape == (2, 2)
    assert logits.device.type == "cpu"
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
