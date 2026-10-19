"""Small real CPU architecture, learning and weight-compatibility probes."""

import pytest
import torch
from torch import nn

from hyso.models.modular import ModularCNN
from hyso.models.blocks import ConvNormAct, ResidualBlock
from hyso.models.bcam import CBAMBlock
from hyso.models.WithSE import SEBlock
from hyso.models.attention import ECAAttention, CoordinateAttention


ATTENTIONS = ["none", "se", "cbam", "eca", "coordinate"]


@pytest.mark.parametrize("widths,repeats,strides", [
    ((4,), (2,), (1,)), ((4, 6, 8), (2, 1, 2), (1, 2, 2)),
    ((4, 4), (1, 2), (2, 1)),
])
@pytest.mark.parametrize("input_channels,num_classes", [(1, 1), (3, 2), (5, 7)])
@pytest.mark.parametrize("batch", [1, 2])
def test_architecture_stage_shapes(widths, repeats, strides, input_channels, num_classes, batch):
    model = ModularCNN(input_channels, num_classes, widths, repeats, strides, norm=None).eval()
    assert isinstance(model.stem, ConvNormAct)
    assert isinstance(model.pool, nn.AdaptiveAvgPool2d) and isinstance(model.classifier, nn.Linear)
    assert len(model.stages) == len(widths)
    seen = []
    def record(module, args, out):
        seen.append(tuple(out.shape))
    handles = [stage.register_forward_hook(record) for stage in model.stages]
    x = torch.randn(batch, input_channels, 11, 15)
    try:
        with torch.no_grad():
            features = model.forward_features(x)
    finally:
        for handle in handles:
            handle.remove()
    height, width = 11, 15
    for index, (channels, repeat, stride) in enumerate(zip(widths, repeats, strides)):
        stage = model.stages[index]
        assert len(stage) == repeat and all(isinstance(block, ResidualBlock) for block in stage)
        height, width = (height + stride - 1) // stride, (width + stride - 1) // stride
        assert seen[index] == (batch, channels, height, width)
        assert stage[0].conv1.conv.stride == (stride, stride)
        assert all(block.conv1.conv.stride == (1, 1) for block in stage[1:])
    assert features.shape == (batch, widths[-1], height, width)
    with torch.no_grad():
        logits = model(x)
        assert logits.shape == (batch, num_classes) and torch.isfinite(logits).all()


@pytest.mark.parametrize("attention", ATTENTIONS)
@pytest.mark.parametrize("norm", ["batch", "group", None])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_attention_learning_and_instance_independence(attention, norm, dtype):
    model = ModularCNN(num_classes=3, widths=(4, 6), repeats=(2, 1), strides=(1, 2),
                       attention=attention, norm=norm).to(dtype)
    blocks = [block for stage in model.stages for block in stage]
    expected = dict(none=nn.Identity, se=SEBlock, cbam=CBAMBlock,
                    eca=ECAAttention, coordinate=CoordinateAttention)[attention]
    assert all(type(block.attention) is expected for block in blocks)
    assert len({id(block.attention) for block in blocks}) == len(blocks)
    weights = [p for block in blocks for p in block.attention.parameters()]
    assert len({id(p) for p in weights}) == len(weights)
    assert all(any(p is registered for registered in model.parameters()) for p in weights)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    identities = [(name, id(p)) for name, p in model.named_parameters()]
    x = torch.randn(2, 3, 9, 13, dtype=dtype, requires_grad=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    logits = model(x)
    assert logits.shape == (2, 3) and logits.dtype == dtype and torch.isfinite(logits).all()
    nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad) > 0
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g) > 0 for g in grads)
    optimizer.step()
    assert any(not torch.equal(before[name], p) for name, p in model.named_parameters())
    if attention != "none":
        for stage_index, stage in enumerate(model.stages):
            for block_index, block in enumerate(stage):
                prefix = f'stages.{stage_index}.{block_index}.attention.'
                assert any(not torch.equal(before[prefix + name], p)
                           for name, p in block.attention.named_parameters())
    assert identities == [(name, id(p)) for name, p in model.named_parameters()]
    assert not any(hasattr(model, name) for name in ('fit', 'optimizer', '_callbacks', 'resume_from'))


@pytest.mark.parametrize("attention", ATTENTIONS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_trained_serialized_strict_round_trip(attention, dtype, tmp_path):
    options = dict(input_channels=1, num_classes=2, widths=(4, 6), repeats=(1, 1),
                   strides=(1, 2), attention=attention)
    model = ModularCNN(**options).to(dtype)
    x = torch.randn(2, 1, 9, 11, dtype=dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    model(x).square().mean().backward()
    optimizer.step()
    buffers = dict(model.named_buffers())
    assert any(name.endswith('num_batches_tracked') and value.item() == 1
               for name, value in buffers.items())
    path = tmp_path/'modular.pth'
    torch.save(model.state_dict(), path)
    fresh = ModularCNN(**options).to(dtype)
    assert any(not torch.equal(p, q) for p, q in zip(model.parameters(), fresh.parameters()))
    result = fresh.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    assert list(fresh.state_dict()) == list(model.state_dict())
    for name, tensor in model.state_dict().items():
        assert fresh.state_dict()[name].shape == tensor.shape
        torch.testing.assert_close(fresh.state_dict()[name], tensor, rtol=0, atol=0)
    model.eval(); fresh.eval()
    before = {name: value.clone() for name, value in model.named_buffers()}
    with torch.no_grad():
        expected = model(x)
        torch.testing.assert_close(fresh(x), expected, rtol=0, atol=0)
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    assert all(torch.equal(before[name], value) for name, value in model.named_buffers())


def test_raw_logits_and_forward_features_contract():
    model = ModularCNN(num_classes=2, widths=(4,), repeats=(1,), strides=(1,)).eval()
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([-2., 3.]))
        x = torch.randn(2, 3, 5, 7)
        assert model.forward_features(x).shape == (2, 4, 5, 7)
        torch.testing.assert_close(model(x), torch.tensor([[-2., 3.], [-2., 3.]]), rtol=0, atol=0)


def test_lists_are_snapshotted_without_numeric_coercion():
    widths, repeats, strides = [4, 6], [1, 2], [1, 2]
    model = ModularCNN(widths=widths, repeats=repeats, strides=strides)
    widths[0] = 99; repeats.clear(); strides[1] = 1
    assert model.widths == (4, 6) and model.repeats == (1, 2) and model.strides == (1, 2)
    assert model.stem.conv.out_channels == 4 and len(model.stages[1]) == 2


@pytest.mark.parametrize("overrides", [
    {'input_channels': 0}, {'input_channels': True}, {'input_channels': 1.5},
    {'num_classes': -1}, {'num_classes': False}, {'num_classes': '2'},
    {'widths': ()}, {'repeats': ()}, {'strides': ()},
    {'widths': [4, 0, 8]}, {'widths': (4, False, 8)}, {'widths': (4., 6, 8)},
    {'widths': '4,6,8'}, {'widths': 4}, {'widths': {4, 6, 8}},
    {'repeats': (1, -1, 1)}, {'repeats': (1, True, 1)}, {'repeats': (1, .5, 1)},
    {'repeats': None}, {'strides': (1, 0, 2)}, {'strides': (1, 2., 2)},
    {'strides': (1, True, 2)}, {'strides': (1, (2, 1), 2)},
    {'widths': (4,)}, {'repeats': (1,)}, {'strides': (1,)},
    {'attention': 'SE'}, {'attention': None}, {'attention': nn.Identity()},
    {'norm': 'auto'}, {'norm': False}, {'norm': nn.BatchNorm2d},
])
def test_invalid_config_rejected(overrides):
    with pytest.raises(ValueError):
        ModularCNN(**overrides)


@pytest.mark.parametrize("shape", [(2, 4, 5, 7), (2, 3, 5), (1, 3, 0, 4), (0, 3, 5, 7)])
def test_invalid_input_shape(shape):
    model = ModularCNN(widths=(4,), repeats=(1,), strides=(1,))
    with pytest.raises(ValueError, match='input|NCHW'):
        model(torch.randn(*shape))


@pytest.mark.parametrize("change", [dict(widths=(4, 8)), dict(num_classes=3),
                                    dict(input_channels=1), dict(repeats=(1, 2)),
                                    dict(attention='se'), dict(norm='group')])
def test_incompatible_weight_schemas_rejected(change):
    options = dict(widths=(4, 6), repeats=(1, 1), strides=(1, 2))
    source = ModularCNN(**options)
    options.update(change)
    target = ModularCNN(**options)
    with pytest.raises(RuntimeError):
        target.load_state_dict(source.state_dict(), strict=True)
