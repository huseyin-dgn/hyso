"""Deterministic CPU mobile-operator and deployment-conversion contracts."""
import copy
import io
import json
import os
from pathlib import Path

import pytest

# Historical families remain available only in the development checkout profile.
pytestmark = pytest.mark.development_profile
import torch
from torch import nn
from torch.nn import functional as F

from hyso import ModelBuilder, ModelRegistry
from hyso.models import (InvertedResidual, MBConv, FusedMBConv, GhostConv, GhostBottleneck,
    ChannelShuffle, ShuffleUnit, DropBlock2D, RepVGGBlock, MobileOneBlock,
    reparameterize_model)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('M4 CPU test attempted CUDA work')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    for name in ('_lazy_init','current_device','synchronize'):
        monkeypatch.setattr(torch.cuda,name,forbidden)


BLOCKS = [
    ('inverted_residual',dict(in_channels=4,out_channels=8,stride=2,expand_ratio=2.)),
    ('mbconv',dict(in_channels=4,out_channels=8,stride=2,expand_ratio=2.)),
    ('fused_mbconv',dict(in_channels=4,out_channels=8,stride=2,expand_ratio=2.)),
    ('ghost_conv',dict(in_channels=4,out_channels=7,stride=2,ratio=3)),
    ('ghost_bottleneck',dict(in_channels=4,out_channels=8,hidden_channels=12,stride=2,se_ratio=.25)),
    ('channel_shuffle',dict(channels=4,groups=2)),
    ('shuffle_unit',dict(in_channels=4,out_channels=8,stride=2)),
    ('repvgg_block',dict(in_channels=4,out_channels=8,stride=2,groups=2)),
    ('mobileone_block',dict(in_channels=4,out_channels=8,stride=2,groups=2,num_conv_branches=3,use_se=True)),
    ('dropblock2d',dict(p=.2,block_size=3)),
]


@pytest.mark.parametrize('name,params',BLOCKS)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_blocks_builder_gradient_state_and_parameter_bound(name,params,dtype):
    from hyso.models.m4_registry import block_cost
    registry = ModelRegistry()
    normalized = registry.validate(name,params)
    block = registry.create(name,**params).to(dtype)
    model = ModelBuilder((4,9,11),3).add(name,**params).build().to(dtype)
    x = torch.randn(2,4,9,11,dtype=dtype,requires_grad=True)
    y = block(x)
    stride = params.get('stride',1)
    assert y.shape == (2,params.get('out_channels',4),(9+stride-1)//stride,(11+stride-1)//stride)
    assert y.dtype == dtype and y.device.type == 'cpu' and torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    assert block_cost(name,normalized) >= sum(p.numel() for p in block.parameters())
    assert model(torch.randn(2,4,9,11,dtype=dtype)).shape == (2,3)
    target = registry.create(name,**params).to(dtype)
    stream = io.BytesIO()
    torch.save(block.state_dict(),stream)
    stream.seek(0)
    target.load_state_dict(torch.load(stream,weights_only=True),strict=True)
    block.eval()
    target.eval()
    assert torch.equal(block(x),target(x))


def test_linear_bottleneck_residual_and_channel_shuffle_reference():
    block = InvertedResidual(4,4,expand_ratio=1.)
    assert block.use_res_connect and type(block.conv[-2]) is nn.Conv2d
    for p in block.conv.parameters():
        nn.init.zeros_(p)
    x = torch.randn(2,4,3,5)
    assert torch.equal(block(x),x)
    assert not InvertedResidual(4,8,expand_ratio=2.).use_res_connect
    ordered = torch.arange(6.).reshape(1,6,1,1)
    assert ChannelShuffle(6,2)(ordered).flatten().tolist() == [0,3,1,4,2,5]
    unit = ShuffleUnit(4,4).eval()
    first,second = x.chunk(2,1)
    expected = ChannelShuffle(4)(torch.cat((first,unit.branch2(second)),1))
    assert torch.equal(unit(x),expected)


def test_ghost_cheap_features():
    x = torch.randn(2,4,5,7,requires_grad=True)
    original = x.detach().clone()
    ghost = GhostConv(4,7,ratio=3).eval()
    primary = ghost.primary(x)
    assert ghost.cheap[0].groups == 3
    assert torch.equal(ghost(x),torch.cat((primary,ghost.cheap(primary)),1)[:,:7])


def test_inverted_residual_near_one_expansion_parameter_bound():
    from hyso.models.m4_registry import block_cost
    params = ModelRegistry().validate('inverted_residual',
                                     dict(in_channels=4, out_channels=4, expand_ratio=1.0001))
    block = ModelRegistry().create('inverted_residual', **params)
    assert block_cost('inverted_residual', params) == sum(p.numel() for p in block.parameters())


@pytest.mark.parametrize('shape',[(2,3,9,11),(2,3,2,4),(1,1,1,1),(1,1,4,4)])
@pytest.mark.parametrize('size',[1,2,3,15])
def test_dropblock_small_shapes_and_finite_normalization(shape,size):
    model = DropBlock2D(.7,size)
    x = torch.ones(shape,requires_grad=True)
    torch.random.default_generator.manual_seed(13)
    y = model(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    model.eval()
    assert model(x) is x
    assert DropBlock2D(0,size)(x) is x
    assert torch.count_nonzero(DropBlock2D(1,size)(x)) == 0


def test_dropblock_reference_contiguous_mask_and_half_all_dropped(monkeypatch):
    def random(shape,device=None):
        values = torch.ones(shape,device=device)
        values[...,1,2] = 0
        return values
    monkeypatch.setattr(torch,'rand',random)
    output = DropBlock2D(.5,3)(torch.ones(1,1,5,6))
    assert torch.equal(output[0,0,1:4,2:5],torch.zeros(3,3))
    assert torch.count_nonzero(output == 0) == 9
    assert torch.allclose(output.mean(),torch.tensor(1.))
    monkeypatch.setattr(torch,'rand',lambda shape,device=None:torch.zeros(shape,device=device))
    x = torch.ones(1,1,260,260,dtype=torch.float16,requires_grad=True)
    y = DropBlock2D(.9,1)(x)
    y.sum().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all() and torch.count_nonzero(y) == 0


@pytest.fixture(scope='module')
def fusion_metrics():
    records = []
    yield records
    destination = os.environ.get('HYSO_M4_FUSION_METRICS')
    if destination:
        Path(destination).write_text(json.dumps(records,indent=2),encoding='utf-8')


def random_bn(model):
    with torch.no_grad():
        for bn in model.modules():
            if type(bn) is nn.BatchNorm2d:
                bn.running_mean.uniform_(-.4,.4)
                bn.running_var.uniform_(.2,1.7)
                if bn.affine:
                    bn.weight.uniform_(.3,1.5)
                    bn.bias.uniform_(-.2,.2)
                bn.eps = .003


@pytest.mark.parametrize('kind',['rep','one3','one1','one5'])
@pytest.mark.parametrize('groups,stride,output',[(1,1,4),(2,1,4),(4,1,4),(2,2,8)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_fusion_nontrivial_bn_groups_stride_and_roundtrip(kind,groups,stride,output,dtype,fusion_metrics):
    if kind == 'rep':
        block = RepVGGBlock(4,output,stride,groups)
    else:
        kernel = int(kind[-1])
        block = MobileOneBlock(4,output,kernel,stride,groups,num_conv_branches=3,use_se=True)
    block = block.to(dtype).eval()
    random_bn(block)
    x = torch.randn(2,4,9,13,dtype=dtype)
    with torch.no_grad():
        reference = block(x)
        converted = reparameterize_model(block)
        actual = converted(x)
    tolerance = dict(atol=2e-6,rtol=2e-5) if dtype == torch.float32 else dict(atol=3e-12,rtol=3e-12)
    torch.testing.assert_close(actual,reference,**tolerance)
    difference = (actual-reference).abs().max().item()
    fusion_metrics.append(dict(kind=kind,dtype=str(dtype),groups=groups,stride=stride,
        max_abs=difference,relative_linf=difference/max(reference.abs().max().item(),1e-12),**tolerance))
    assert not block.deploy and converted.deploy
    assert converted.reparam.weight.device == x.device and converted.reparam.weight.dtype == dtype
    parameter_id = id(converted.reparam.weight)
    converted.switch_to_deploy()
    assert id(converted.reparam.weight) == parameter_id and torch.equal(converted(x),actual)
    if kind == 'rep':
        fresh = RepVGGBlock(4,output,stride,groups,deploy=True).to(dtype).eval()
    else:
        fresh = MobileOneBlock(4,output,kernel,stride,groups,num_conv_branches=3,use_se=True,deploy=True).to(dtype).eval()
    stream = io.BytesIO()
    torch.save(converted.state_dict(),stream)
    stream.seek(0)
    fresh.load_state_dict(torch.load(stream,weights_only=True),strict=True)
    assert torch.equal(fresh(x),actual)
    for strict in (True,False):
        with pytest.raises(RuntimeError,match='topology'):
            block.load_state_dict(fresh.state_dict(),strict=strict)
        with pytest.raises(RuntimeError,match='topology'):
            fresh.load_state_dict(block.state_dict(),strict=strict)


def test_conv_bias_affine_false_and_identity_kernel_reference():
    block = RepVGGBlock(4,4,groups=2).double().eval()
    block.identity = nn.BatchNorm2d(4,affine=False).double().eval()
    block.branches[0][0].bias = nn.Parameter(torch.randn(4,dtype=torch.float64))
    random_bn(block)
    x = torch.randn(2,4,7,9,dtype=torch.float64)
    expected = block(x)
    block.switch_to_deploy()
    torch.testing.assert_close(block(x),expected,atol=3e-12,rtol=3e-12)


@pytest.mark.parametrize('fault',['training','running_stats','variance','kernel','half'])
def test_conversion_rejects_unsupported_before_mutation(fault):
    first = RepVGGBlock(4,4)
    second = MobileOneBlock(4,4)
    model = nn.Sequential(first,second).eval()
    if fault == 'training':
        second.train()
    elif fault == 'running_stats':
        second.branches[0][1] = nn.BatchNorm2d(4,track_running_stats=False).eval()
    elif fault == 'variance':
        second.branches[0][1].running_var[0] = -1
    elif fault == 'kernel':
        second.branches[0][0].padding = (0,0)
    else:
        second.half()
    with pytest.raises(ValueError):
        reparameterize_model(model,inplace=True)
    assert not first.deploy and not second.deploy and hasattr(first,'branches')


def test_builder_converted_graph_exports_the_current_phase():
    model = ModelBuilder((4,7,9),3).add('repvgg_block',in_channels=4,out_channels=4).add(
        'mobileone_block',in_channels=4,out_channels=8,stride=2).build().eval()
    x = torch.randn(2,4,7,9)
    expected = model(x)
    deployed = reparameterize_model(model)
    config = deployed.to_config()
    assert all(entry['params']['deploy'] for entry in config['blocks'])
    fresh = ModelBuilder.from_config(config).build().eval()
    fresh.load_state_dict(deployed.state_dict(),strict=True)
    torch.testing.assert_close(fresh(x),expected,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('name,params',[
    ('inverted_residual',dict(in_channels=4,out_channels=4,stride=3)),
    ('mbconv',dict(in_channels=4,out_channels=4,expand_ratio=0)),
    ('fused_mbconv',dict(in_channels=4,out_channels=4,kernel_size=2)),
    ('ghost_conv',dict(in_channels=4,out_channels=4,ratio=1)),
    ('ghost_bottleneck',dict(in_channels=4,out_channels=4,hidden_channels=0)),
    ('channel_shuffle',dict(channels=5,groups=2)),
    ('shuffle_unit',dict(in_channels=4,out_channels=7,stride=2)),
    ('shuffle_unit',dict(in_channels=4,out_channels=8)),
    ('repvgg_block',dict(in_channels=4,out_channels=6,groups=4)),
    ('mobileone_block',dict(in_channels=4,out_channels=4,num_conv_branches=0)),
    ('dropblock2d',dict(p=float('nan'))),('dropblock2d',dict(block_size=0)),
])
def test_invalid_registry_block_params(name,params):
    with pytest.raises(ValueError):
        ModelRegistry().validate(name,params)
