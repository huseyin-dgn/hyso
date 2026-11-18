"""Small real CPU probes for mechanisms, gradients, buffers and weight loading."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from hyso.models.attention import ECAAttention, CoordinateAttention


def assert_backward(module, x, out):
    assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0
    grads = [p.grad for p in module.parameters()]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g) > 0 for g in grads)


@pytest.mark.parametrize("channels", [1, 3, 8, 16])
@pytest.mark.parametrize("shape", [(1, 5, 7), (2, 3, 5), (1, 1, 7), (2, 5, 1), (1, 1, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_eca_shapes_backward_and_registration(channels, shape, dtype):
    batch, height, width = shape
    module = ECAAttention(channels).to(dtype)
    before = [(n, id(p)) for n, p in module.named_parameters()]
    x = torch.randn(batch, channels, height, width, dtype=dtype, requires_grad=True)
    assert_backward(module, x, module(x))
    assert before == [(n, id(p)) for n, p in module.named_parameters()]
    assert sum(p.numel() for p in module.parameters()) == 3  # No channel-reduction MLP.


@pytest.mark.parametrize("kernel", [1, 3, 5, 9])
def test_eca_channel_axis_formula(kernel):
    module = ECAAttention(3, kernel_size=kernel).double()
    x = (torch.arange(2 * 3 * 3 * 5, dtype=torch.float64).reshape(2, 3, 3, 5) - 20) / 100
    with torch.no_grad():
        module.conv.weight.copy_(torch.linspace(-.3, .4, kernel, dtype=torch.float64).reshape(1, 1, kernel))
    # Scalar channel-neighborhood reference: no production pooling/transpose/Conv1d calls.
    gates = torch.empty(2, 3, dtype=torch.float64)
    for b in range(2):
        for c in range(3):
            value = 0.
            for tap in range(kernel):
                neighbor = c + tap - kernel // 2
                if 0 <= neighbor < 3:
                    value += module.conv.weight[0, 0, tap] * x[b, neighbor].mean()
            gates[b, c] = torch.sigmoid(value)
    torch.testing.assert_close(module(x), x * gates[:, :, None, None], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("channels,reduction", [(1, 32), (3, 1), (8, 2), (16, 1)])
@pytest.mark.parametrize("shape", [(1, 5, 7), (2, 3, 5), (1, 1, 7), (1, 5, 1), (1, 1, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_coordinate_shapes_backward_bn_and_registration(channels, reduction, shape, dtype):
    batch, height, width = shape
    module = CoordinateAttention(channels, reduction).to(dtype)
    assert module.conv1.out_channels == max(8, channels // reduction)
    before = [(n, id(p)) for n, p in module.named_parameters()]
    x = torch.randn(batch, channels, height, width, dtype=dtype, requires_grad=True)
    old_var = module.bn.running_var.clone()
    assert_backward(module, x, module(x))
    assert module.bn.num_batches_tracked.item() == 1
    assert not torch.equal(old_var, module.bn.running_var)
    assert torch.isfinite(module.bn.running_mean).all() and torch.isfinite(module.bn.running_var).all()
    assert before == [(n, id(p)) for n, p in module.named_parameters()]


def test_coordinate_axis_pooling_split_gates_and_reference_formula():
    module = CoordinateAttention(3, reduction=2).double().eval()
    x = torch.arange(2 * 3 * 3 * 5, dtype=torch.float64).reshape(2, 3, 3, 5) / 100
    seen = {}
    def capture(name):
        def hook(layer, args, output):
            seen[name] = (args[0].detach().clone(), output.detach().clone())
        return hook
    handles = [getattr(module, name).register_forward_hook(capture(name))
               for name in ("pool_h", "pool_w", "conv1", "bn", "conv_h", "conv_w")]
    try:
        actual = module(x)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(seen['pool_h'][1], x.mean(3, keepdim=True), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(seen['pool_w'][1], x.mean(2, keepdim=True), rtol=1e-12, atol=1e-12)
    assert seen['conv1'][0].shape == (2, 3, 8, 1)
    torch.testing.assert_close(seen['conv1'][0][:, :, :3], x.mean(3, keepdim=True), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(seen['conv1'][0][:, :, 3:], x.mean(2, keepdim=True).transpose(2, 3), rtol=1e-12, atol=1e-12)
    transformed = seen['bn'][1] * F.relu6(seen['bn'][1] + 3) / 6  # Original h-swish definition.
    torch.testing.assert_close(seen['conv_h'][0], transformed[:, :, :3], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(seen['conv_w'][0], transformed[:, :, 3:].transpose(2, 3), rtol=1e-12, atol=1e-12)
    gate_h, gate_w = seen['conv_h'][1].sigmoid(), seen['conv_w'][1].sigmoid()
    assert gate_h.shape == (2, 3, 3, 1) and gate_w.shape == (2, 3, 1, 5)
    torch.testing.assert_close(actual, x * gate_w * gate_h, rtol=0, atol=0)


@pytest.mark.parametrize("kind", [ECAAttention, CoordinateAttention])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_attention_train_eval_strict_serialized_round_trip(kind, dtype, tmp_path):
    source = kind(3).to(dtype)
    x = torch.randn(1, 3, 3, 5, dtype=dtype)
    optimizer = torch.optim.Adam(source.parameters(), lr=.01)
    source(x).square().mean().backward()
    optimizer.step()
    path = tmp_path/'attention.pth'
    torch.save(source.state_dict(), path)
    target = kind(3).to(dtype)
    result = target.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    assert list(target.state_dict()) == list(source.state_dict())
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
    source.eval(); target.eval()
    before = {n: b.clone() for n, b in source.named_buffers()}
    with torch.no_grad():
        expected = source(x)
        torch.testing.assert_close(source(x), expected, rtol=0, atol=0)
        torch.testing.assert_close(target(x), expected, rtol=0, atol=0)
    assert all(torch.equal(before[n], b) for n, b in source.named_buffers())


@pytest.mark.parametrize("kind", [ECAAttention, CoordinateAttention])
@pytest.mark.parametrize("channels", [0, -1, True, 1.5, "3"])
def test_attention_invalid_channels(kind, channels):
    with pytest.raises(ValueError, match="channels"):
        kind(channels)


@pytest.mark.parametrize("kernel", [0, -1, 2, 4, True, 1.5, (3,)])
def test_eca_invalid_kernel(kernel):
    with pytest.raises(ValueError, match="kernel_size"):
        ECAAttention(3, kernel)


@pytest.mark.parametrize("reduction", [0, -1, True, .5, "8"])
def test_coordinate_invalid_reduction(reduction):
    with pytest.raises(ValueError, match="reduction"):
        CoordinateAttention(3, reduction)


@pytest.mark.parametrize("kind", [ECAAttention, CoordinateAttention])
@pytest.mark.parametrize("shape", [(2, 3, 5), (1, 4, 3, 5), (1, 3, 0, 5), (0, 3, 2, 2)])
def test_attention_invalid_input_shape(kind, shape):
    with pytest.raises(ValueError, match="NCHW|channels|nonempty"):
        kind(3)(torch.randn(*shape))
