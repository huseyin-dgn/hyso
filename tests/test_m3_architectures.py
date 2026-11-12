"""Short deterministic CPU shape, topology, derivative and integration checks."""
import io
import json
import gc
from pathlib import Path
import pytest
import torch
from torch import nn
from hyso import ModelBuilder, ModelRegistry, TrainingConfig
from hyso.models import (ResNet, DenseNet, ResNeXt, PreActResNet, WideResNet,
    BasicResidualBlock, Bottleneck, ResNeXtBottleneck, PreActivationBlock, WideResidualBlock,
    SimAM, TripletAttention, SelectiveKernelAttention)

VARIANTS=[('resnet',d,{}) for d in (18,34,50,101,152)]+[
    ('densenet',d,{}) for d in (121,169,201)]+[
    ('preact_resnet',d,{'base_channels':4}) for d in (18,34,50,101,152)]+[
    ('wide_resnet',10,{'widen_factor':1}),('wide_resnet',28,{'widen_factor':2}),
    ('resnext',50,{}),('resnext',101,{}),('resnext',101,{'width_per_group':8}),
    ('wide_resnet',28,{'widen_factor':10}),('preact_resnet',18,{})]


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('M3 CPU test attempted CUDA access')
    # AdamW's CPU step asks this metadata question for graph-capture health.
    # It is not a GPU workload; keep CPU-only availability while fencing actual
    # context initialization/device work. Models, derivatives and fit remain real.
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    for name in ('_lazy_init','device_count','current_device','synchronize'):
        monkeypatch.setattr(torch.cuda,name,forbidden)


@pytest.mark.parametrize('name,depth,options',VARIANTS)
def test_every_depth_forward(name,depth,options):
    model=ModelRegistry().create(name,depth=depth,input_channels=1,num_classes=3,**options).eval()
    with torch.inference_mode(): y=model(torch.randn(1,1,32,32))
    assert y.shape==(1,3) and torch.isfinite(y).all()
    del model;gc.collect()


def test_canonical_topology_and_state_key_equivalence():
    from torchvision.models import resnet18,densenet121
    for ctor,ref in ((ResNet,resnet18),(DenseNet,densenet121)):
        model=ctor();reference=ref(weights=None)
        assert [(k,v.shape) for k,v in model.state_dict().items()]==[(k,v.shape) for k,v in reference.state_dict().items()]
        model.load_state_dict(reference.state_dict(),strict=True)
        del model,reference;gc.collect()
    model=ResNet(depth=50)
    assert model.layer2[0].conv2.stride==(2,2) and model.layer2[0].conv1.stride==(1,1)
    assert model.layer4[-1].expansion==4
    del model


@pytest.mark.parametrize('ctor,options',[(BasicResidualBlock,{}),(Bottleneck,{}),(ResNeXtBottleneck,{'groups':2,'width_per_group':32}),
    (PreActivationBlock,{}),(PreActivationBlock,{'bottleneck':True}),(WideResidualBlock,{'dropout':.1})])
@pytest.mark.parametrize('stride',[1,2])
def test_projection_and_backward(ctor,options,stride):
    block=ctor(4,8,stride=stride,**options).double()
    x=torch.randn(2,4,7,9,dtype=torch.float64,requires_grad=True)
    y=block(x);assert y.shape==(2,8,(7+stride-1)//stride,(9+stride-1)//stride)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())


def test_preact_identity_has_no_post_add_activation():
    model=PreActivationBlock(4,4,norm=None)
    for p in model.main.parameters():nn.init.zeros_(p)
    x=-torch.ones(2,4,3,3)
    assert torch.equal(model(x),x)


@pytest.mark.parametrize('ctor',[SimAM,TripletAttention,SelectiveKernelAttention])
@pytest.mark.parametrize('shape',[(2,4,5,7),(2,4,1,1)])
def test_attention_gradient_and_serialization(ctor,shape):
    model=ctor(4).double();x=torch.randn(*shape,dtype=torch.float64,requires_grad=True)
    y=model(x);assert y.shape==x.shape and torch.isfinite(y).all()
    y.square().mean().backward();assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    clone=ctor(4).double();clone.load_state_dict(model.state_dict(),strict=True)
    model.eval();clone.eval();assert torch.equal(model(x),clone(x))
    with pytest.raises(ValueError):model(torch.randn(2,3,4,4,dtype=torch.float64))
    with pytest.raises(ValueError):model(torch.ones(2,4,4,4,dtype=torch.int64))


def test_simam_formula_gradcheck_and_extreme_values():
    x=torch.randn(1,2,2,3,dtype=torch.float64,requires_grad=True)
    m=SimAM(2)
    d=(x-x.mean((2,3),keepdim=True)).square()
    expected=x*torch.sigmoid(d/(4*(d.sum((2,3),keepdim=True)/5+1e-4))+.5)
    assert torch.allclose(m(x),expected,atol=1e-12)
    assert torch.autograd.gradcheck(m,(x,))
    for dtype in (torch.float32,torch.float16,torch.bfloat16):
        v=torch.full((2,2,1,1),10000.,dtype=dtype,requires_grad=True)
        m(v).float().sum().backward();assert torch.isfinite(v.grad).all()


@pytest.mark.parametrize('name,params',[
 ('resnet',{'depth':20}),('resnext',{'depth':18}),('densenet',{'depth':161}),
 ('wide_resnet',{'depth':12}),('wide_resnet',{'dropout':1.}),('preact_resnet',{'base_channels':0}),
 ('simam',{'channels':4,'epsilon':0}),('triplet',{'channels':4,'kernel_size':2}),
 ('selective_kernel',{'channels':4,'groups':3}),('bottleneck',{'in_channels':4,'out_channels':5}),
 ('basic_residual',{'in_channels':4,'out_channels':8,'stride':3})])
def test_invalid_params_fail_before_build(name,params):
    with pytest.raises(ValueError):ModelRegistry().validate(name,params)


@pytest.mark.parametrize('name,params',[
 ('resnet',{'depth':18}),('resnext',{'depth':50}),('densenet',{'depth':121,'growth_rate':4}),
 ('preact_resnet',{'depth':18,'base_channels':4}),('wide_resnet',{'depth':10,'widen_factor':1})])
def test_builder_json_and_real_state_dict_roundtrip(name,params,tmp_path):
    builder=ModelBuilder.named(name,input_channels=2,num_classes=5,**params)
    p=tmp_path/'architecture.json';builder.to_json(p)
    restored=ModelBuilder.from_json(p);assert restored.to_config()==builder.to_config()
    model=builder.build().eval();clone=restored.build().eval()
    stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
    clone.load_state_dict(torch.load(stream,weights_only=True),strict=True)
    with torch.inference_mode():
        x=torch.randn(1,2,32,32);assert torch.equal(model(x),clone(x))


@pytest.mark.parametrize('attention',['simam','triplet','selective_kernel'])
def test_custom_builder_and_legacy_residual_attention(attention):
    model=ModelBuilder((2,8,8),3).add('basic_residual',in_channels=2,out_channels=4).add(attention,channels=4).build()
    assert model(torch.randn(2,2,8,8)).shape==(2,3)
    legacy=ModelRegistry().create('residual',in_channels=2,out_channels=4,attention=dict(name=attention,params={'channels':4}))
    assert legacy(torch.randn(2,2,8,8)).shape==(2,4,8,8)


@pytest.mark.parametrize('name,params',[('preact_resnet',{'base_channels':4}),('wide_resnet',{'depth':10,'widen_factor':1}),('densenet',{'growth_rate':4})])
def test_training_session_short_real_step(name,params):
    from torch.utils.data import DataLoader,TensorDataset
    model=ModelBuilder.named(name,input_channels=1,num_classes=2,**params).build()
    loader=DataLoader(TensorDataset(torch.randn(2,1,32,32),torch.tensor([0,1])),batch_size=2)
    session=TrainingConfig(epochs=1,batch_size=2).prepare(model,loader)
    before=next(model.parameters()).detach().clone();result=session.fit()
    assert result.to_dict()['applied_updates']==1
    assert not torch.equal(before,next(model.parameters()))
    assert session.settings()['state']=='completed'


def test_cli_discovery_validate_summary_without_model_construction(tmp_path,capsys,monkeypatch):
    from hyso.cli import main
    from hyso.models import architectures
    p=tmp_path/'model.json';ModelBuilder.named('resnet',depth=152).to_json(p)
    monkeypatch.setattr(architectures.ResNet,'__init__',lambda *a,**k:pytest.fail('dry metadata constructed model'))
    for args in [['models','list'],['models','show','resnet'],['models','validate',str(p)],['models','summary',str(p)]]:
        assert main([*args,'--json'])==0
        assert json.loads(capsys.readouterr().out)['exit_code']==0


@pytest.mark.parametrize('norm',['group',None])
@pytest.mark.parametrize('name',['basic_residual','bottleneck','preactivation','wide_residual','resnext_bottleneck'])
def test_block_norms_and_builder_budget(name,norm):
    params=dict(in_channels=4,out_channels=16,stride=2,norm=norm)
    if name=='resnext_bottleneck':params.update(groups=2,width_per_group=32)
    model=ModelBuilder((4,7,9),3).add(name,**params).build()
    y=model(torch.randn(2,4,7,9));assert y.shape==(2,3)
    y.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('name,params',[('resnet',{'depth':18}),('resnext',{'depth':50})])
def test_large_family_session_prepare_without_large_backward(name,params):
    from torch.utils.data import DataLoader,TensorDataset
    model=ModelBuilder.named(name,input_channels=1,num_classes=2,**params).build()
    loader=DataLoader(TensorDataset(torch.ones(2,1,32,32),torch.tensor([0,1])),batch_size=2)
    session=TrainingConfig(epochs=1,batch_size=2).prepare(model,loader)
    assert session.settings()['state']=='prepared' and session.settings()['effective']['device']=='cpu'


def test_new_family_automatic_checkpoint_resume_remains_closed(tmp_path):
    from torch.utils.data import DataLoader,TensorDataset
    model=PreActResNet(base_channels=4,input_channels=1,num_classes=2)
    loader=DataLoader(TensorDataset(torch.ones(2,1,32,32),torch.tensor([0,1])),batch_size=2)
    with pytest.raises(ValueError):
        TrainingConfig(epochs=1,batch_size=2,save_last_to=str(tmp_path/'last.pth')).prepare(model,loader)
    assert not (tmp_path/'last.pth').exists()


def test_pure_family_budgets_and_no_unknown_json_fields():
    from hyso.models.architectures import validate_family
    assert validate_family('resnext',dict(depth=101,input_channels=3,num_classes=1000,groups=32,width_per_group=8,norm='batch'))<200_000_000
    for name in ('basic_residual','resnext_bottleneck'):
        with pytest.raises(ValueError):ModelBuilder((8192,4,4),2).add(name,in_channels=8192,out_channels=8192)
    with pytest.raises(ValueError):ModelRegistry().validate('simam',dict(channels=4,epsilon=10**1000))
    config=ModelBuilder.named('resnet').to_config();config['import_path']='os:system'
    with pytest.raises(ValueError):ModelBuilder.from_config(config)
    with pytest.raises(ValueError):DenseNet(depth=121.)


@pytest.mark.parametrize('scenario',['resnet','variant','densenet','block','attention','builder','training','state_dict'])
def test_documented_examples(scenario,monkeypatch,capsys):
    import runpy,sys
    path=Path(__file__).resolve().parents[1]/'examples/models/m3_workflows.py'
    monkeypatch.setattr(sys,'argv',[str(path),scenario])
    runpy.run_path(str(path),run_name='__main__')
    assert capsys.readouterr().out.strip()


def test_triplet_rotation_reference():
    from torch.nn import functional as F
    m=TripletAttention(3).double().eval();x=torch.randn(2,3,5,7,dtype=torch.float64)
    def gate(value,module):
        z=torch.cat((value.max(1,keepdim=True).values,value.mean(1,keepdim=True)),1)
        z=F.conv2d(z,module.conv.weight,padding=3)
        n=module.norm
        z=(z-n.running_mean[None,:,None,None])/torch.sqrt(n.running_var[None,:,None,None]+n.eps)
        z=z*n.weight[None,:,None,None]+n.bias[None,:,None,None]
        return value*torch.sigmoid(z)
    expected=(gate(x.permute(0,2,1,3),m.cw).permute(0,2,1,3)+
              gate(x.permute(0,3,2,1),m.hc).permute(0,3,2,1)+gate(x,m.hw))/3
    assert torch.allclose(m(x),expected,atol=1e-12)


def test_sk_selects_over_branches_not_channels():
    import math
    m=SelectiveKernelAttention(4,groups=2).double().eval();x=torch.randn(2,4,5,7,dtype=torch.float64)
    with torch.no_grad():
        for i,head in enumerate(m.select):head.weight.zero_();head.bias.fill_(0 if i==0 else math.log(3))
        expected=.25*m.branches[0](x)+.75*m.branches[1](x)
        assert torch.allclose(m(x),expected,atol=1e-12)
