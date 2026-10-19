"""Real small-tensor tests for new blocks, independent of legacy training."""

import pytest
import torch
from torch import nn

from hyso.models.blocks import ConvNormAct, ResidualBlock
from hyso.models.bcam import CBAMBlock
from hyso.models.WithSE import SEBlock


def assert_gradients(module, inputs):
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert torch.count_nonzero(inputs.grad) > 0
    parameters = [p for p in module.parameters() if p.requires_grad]
    assert parameters and all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    assert any(torch.count_nonzero(p.grad) > 0 for p in parameters)


@pytest.mark.parametrize("norm", ["batch", "group", None])
@pytest.mark.parametrize("activation", ["silu", "relu", None])
@pytest.mark.parametrize("stride,groups", [(1, 1), (2, 2)])
def test_conv_forward_backward(norm, activation, stride, groups):
    block = ConvNormAct(4, 6, stride=stride, groups=groups, norm=norm, activation=activation)
    x = torch.randn(2, 4, 11, 13, requires_grad=True)
    out = block(x)
    assert out.shape == (2, 6, (11 + stride - 1) // stride, (13 + stride - 1) // stride)
    assert out.dtype == x.dtype and out.device == x.device and torch.isfinite(out).all()
    assert (block.conv.bias is not None) == (norm is None)
    out.square().mean().backward()
    assert_gradients(block, x)


def test_conv_explicit_rectangular_kernel_stride_and_padding():
    block = ConvNormAct(4, 8, kernel_size=(3, 5), stride=(2, 1), padding=(0, 2), groups=4)
    assert block(torch.randn(2, 4, 11, 13)).shape == (2, 8, 5, 13)


@pytest.mark.parametrize("channels,groups", [(1, 1), (7, 7), (10, 5), (12, 6), (16, 8)])
def test_group_norm_divisor_policy(channels, groups):
    block = ConvNormAct(2, channels, norm="group")
    assert isinstance(block.norm, nn.GroupNorm)
    assert block.norm.num_groups == groups
    assert block(torch.randn(2, 2, 5, 7)).shape == (2, channels, 5, 7)


@pytest.mark.parametrize("kwargs", [
    {"in_channels": 0}, {"in_channels": True}, {"out_channels": -1},
    {"out_channels": 2.5}, {"groups": 0}, {"groups": True}, {"groups": 3},
    {"groups": 4}, {"kernel_size": 0}, {"kernel_size": (3, 0)},
    {"kernel_size": 1.5}, {"kernel_size": (3,)}, {"stride": -1},
    {"stride": True}, {"stride": (1, 0)}, {"stride": "same"},
    {"padding": -1}, {"padding": (0, -1)}, {"padding": "same"},
    {"norm": "auto"}, {"norm": nn.BatchNorm2d}, {"activation": "gelu"},
    {"activation": False},
])
def test_conv_invalid_configuration(kwargs):
    options = dict(in_channels=4, out_channels=6)
    options.update(kwargs)
    with pytest.raises(ValueError):
        ConvNormAct(**options)


@pytest.mark.parametrize("norm", ["batch", "group", None])
def test_conv_train_eval_and_disabled_activation(norm):
    block = ConvNormAct(2, 4, norm=norm, activation=None)
    x = torch.randn(2, 2, 5, 7) + 2
    if norm == "batch":
        before = block.norm.running_mean.clone()
    block.train()
    actual = block(x)
    # Explicit numerical contract, without updating a second BN running statistic.
    assert isinstance(block.activation, nn.Identity)
    if norm == "batch":
        assert not torch.equal(before, block.norm.running_mean)
        count = block.norm.num_batches_tracked.clone()
    block.eval()
    with torch.no_grad():
        expected = block.norm(block.conv(x))
        torch.testing.assert_close(block(x), expected, rtol=0, atol=0)
        torch.testing.assert_close(block(x), block(x), rtol=0, atol=0)
    if norm == "batch":
        assert torch.equal(count, block.norm.num_batches_tracked)
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def attention_instance(kind, channels):
    return {"none": lambda: None, "identity": nn.Identity,
            "se": lambda: SEBlock(channels, reduction=2),
            "cbam": lambda: CBAMBlock(channels, reduction=2)}[kind]()


@pytest.mark.parametrize("kind", ["none", "identity", "se", "cbam"])
@pytest.mark.parametrize("norm", ["batch", "group", None])
@pytest.mark.parametrize("channels,stride", [(4, 1), (6, 1), (4, 2), (6, (2, 1))])
def test_residual_shapes_gradients_and_update(kind, norm, channels, stride):
    attention = attention_instance(kind, channels)
    block = ResidualBlock(4, channels, stride=stride, norm=norm, attention=attention)
    if attention is not None:
        assert block.attention is attention
    projected = channels != 4 or stride != 1
    assert isinstance(block.shortcut, ConvNormAct if projected else nn.Identity)
    assert isinstance(block.conv2.activation, nn.Identity)
    if projected:
        assert isinstance(block.shortcut.activation, nn.Identity)
    shape = (2, channels, 6 if stride in (2, (2, 1)) else 11, 7 if stride == 2 else 13)
    x = torch.randn(2, 4, 11, 13, requires_grad=True)
    identities = [(name, id(p)) for name, p in block.named_parameters()]
    optimizer = torch.optim.SGD(block.parameters(), lr=.05)
    before = {name: p.detach().clone() for name, p in block.named_parameters()}
    out = block(x)
    assert out.shape == shape and torch.isfinite(out).all()
    out.square().mean().backward()
    assert_gradients(block, x)
    optimizer.step()
    assert any(not torch.equal(before[name], p) for name, p in block.named_parameters())
    assert identities == [(name, id(p)) for name, p in block.named_parameters()]
    assert all(p.device.type == "cpu" for p in block.parameters())


@pytest.mark.parametrize("kind", ["none", "identity", "se", "cbam"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_residual_fresh_strict_round_trip_and_eval_formula(kind, dtype):
    def make():
        return ResidualBlock(3, 6, stride=2, attention=attention_instance(kind, 6)).to(dtype)
    source = make()
    x = torch.randn(2, 3, 9, 11, dtype=dtype, requires_grad=True)
    source(x).square().mean().backward()
    assert_gradients(source, x)
    # Include non-default parameters and trained BN buffers in the round-trip.
    opt = torch.optim.Adam(source.parameters(), lr=.01)
    opt.step()
    target = make().eval()
    assert any(not torch.equal(p, q) for p, q in zip(source.parameters(), target.parameters()))
    loaded = target.load_state_dict(source.state_dict(), strict=True)
    assert loaded.missing_keys == loaded.unexpected_keys == []
    assert list(target.state_dict()) == list(source.state_dict())
    for key, value in source.state_dict().items():
        assert value.dtype == dtype or not value.is_floating_point()
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
    source.eval()
    with torch.no_grad():
        expected = source.activation(source.attention(source.conv2(source.conv1(x))) + source.shortcut(x))
        torch.testing.assert_close(source(x), expected, rtol=0, atol=0)
        actual = target(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.dtype == dtype and actual.device == x.device


def test_attention_registration_and_explicit_sharing():
    shared = SEBlock(4)
    first = ResidualBlock(4, 4, attention=shared)
    second = ResidualBlock(4, 4, attention=shared)
    assert first.attention is second.attention is shared
    assert first.attention.fc1.weight is second.attention.fc1.weight
    independent = ResidualBlock(4, 4, attention=SEBlock(4))
    assert independent.attention.fc1.weight is not shared.fc1.weight
    assert dict(first.named_parameters())["attention.fc1.weight"] is shared.fc1.weight
    first.eval()
    assert not shared.training
    first.double()
    assert shared.fc1.weight.dtype == torch.float64
    # Defaults must not silently share even parameterless modules.
    assert ResidualBlock(4, 4).attention is not ResidualBlock(4, 4).attention


@pytest.mark.parametrize("attention", [SEBlock(3), CBAMBlock(3), CBAMBlock(4, kernel_size=2)])
def test_attention_incompatible_shape_configuration(attention):
    with pytest.raises(ValueError, match="channels|odd spatial"):
        ResidualBlock(4, 4, attention=attention)


@pytest.mark.parametrize("attention", [nn.ReLU(), "se", lambda x: x])
def test_attention_unsupported_type(attention):
    with pytest.raises(TypeError, match="attention must"):
        ResidualBlock(4, 4, attention=attention)


def test_attention_broadcasting_is_rejected_after_caller_mutation():
    attention = nn.Identity()
    attention.register_forward_hook(lambda module, args, output: output[:, :1])
    block = ResidualBlock(4, 4, attention=attention)
    with pytest.raises(ValueError, match="preserve.*shape"):
        block(torch.randn(2, 4, 5, 7))


@pytest.mark.parametrize("options", [
    {"in_channels": 0}, {"out_channels": False}, {"stride": 0},
    {"stride": (2, 0)}, {"norm": "layer"},
])
def test_residual_invalid_configuration(options):
    args = dict(in_channels=4, out_channels=4)
    args.update(options)
    with pytest.raises(ValueError):
        ResidualBlock(**args)
