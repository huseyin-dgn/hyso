"""Bounded CPU validation of original advanced components and their public entry points."""
import copy
import json
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset
from hyso import ModelBuilder, ModelRegistry, ModelConfigError, TrainingConfig
from hyso.cli import main
from hyso.models import advanced

SAMPLES = {name:dict(in_channels=4,out_channels=6,intermediate_channels=8,normalization='group')
           for name in advanced.RESIDUALS-{'inverted_bottleneck'}}
SAMPLES['inverted_bottleneck'] = dict(in_channels=4,out_channels=6,expansion_ratio=2,normalization='group')
SAMPLES.update(spatial_attention=dict(channels=4),channel_attention=dict(channels=4),
               spatial_self_attention=dict(channels=4,embedding_dim=8),
               multihead_spatial_attention=dict(channels=4,embedding_dim=8,num_heads=2),
               dropout=dict(p=.2),dropout2d=dict(p=.2),drop_path=dict(p=.2),gaussian_noise=dict(std=.1))

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    torch.manual_seed(1353)
    def forbidden(*a,**kw): pytest.fail('attempted CUDA query/work')
    for name in ('_lazy_init','is_available','device_count','synchronize'):
        monkeypatch.setattr(torch.cuda,name,forbidden)

def test_registry_metadata_and_closed_legacy_attention():
    r = ModelRegistry()
    assert len(r.list())==76 and set(SAMPLES)==advanced.NEW_NAMES
    for name in SAMPLES:
        meta=r.get(name).to_dict()
        assert meta['description_tr'] and meta['train_eval'] and meta['memory_notes'] and meta['corrections']
        assert json.loads(json.dumps(meta))==meta
    with pytest.raises(ModelConfigError):
        r.validate('residual',dict(in_channels=4,out_channels=4,attention=dict(name='spatial_attention',params=dict(channels=4))))

@pytest.mark.parametrize('name',sorted(SAMPLES))
def test_preflight_without_allocation(name,monkeypatch):
    def forbidden(*a,**kw): pytest.fail('metadata allocated tensors/modules')
    monkeypatch.setattr(nn.Module,'__init__',forbidden)
    monkeypatch.setattr(torch,'empty',forbidden)
    b=ModelBuilder((4,5,7),3).add(name,**SAMPLES[name])
    report=ModelBuilder.preflight(b.to_config(),batch_size=2)
    assert report['valid'] and report['layers'][0]['parameter_count'] is not None

@pytest.mark.parametrize('name',sorted(SAMPLES))
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_json_cpu_modes_gradients_optimizer_and_strict_state(name,dtype,tmp_path):
    b=ModelBuilder((4,5,7),3).add(name,**SAMPLES[name])
    path=tmp_path/'model.json';b.to_json(path)
    clone_b=ModelBuilder.from_json(path)
    assert clone_b.to_config()==b.to_config()
    report=ModelBuilder.preflight(b.to_config())['layers'][0]
    m=b.build().to(dtype)
    assert sum(p.numel() for p in m.features[0].parameters())==report['parameter_count']
    optimizer=torch.optim.SGD(m.parameters(),lr=.01)
    for training in (True,False):
        m.train(training);optimizer.zero_grad(set_to_none=True)
        x=torch.randn(2,4,5,7,dtype=dtype,requires_grad=True)
        assert list(m.features(x).shape)==[2,*report['output_shape'][1:]]
        out=m(x);assert torch.isfinite(out).all()
        out.square().sum().backward()
        assert torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        optimizer.step()
    clone=clone_b.build().to(dtype);clone.load_state_dict(copy.deepcopy(m.state_dict()),strict=True)
    m.eval();clone.eval();x=torch.randn(2,4,5,7,dtype=dtype)
    assert torch.equal(m(x),clone(x))

@pytest.mark.parametrize('name',sorted(advanced.RESIDUALS))
@pytest.mark.parametrize('stride,channels',[(1,4),([2,3],6)])
@pytest.mark.parametrize('norm',[None,'batch','group'])
def test_internal_projection_and_geometry(name,stride,channels,norm):
    p=dict(SAMPLES[name],out_channels=channels,stride=stride,dilation=[2,3],normalization=norm)
    if name=='grouped_bottleneck':p['groups']=4
    b=ModelBuilder((4,9,11),3).add(name,**p)
    detail=ModelBuilder.preflight(b.to_config())['layers'][0]
    m=b.build().features[0].eval();x=torch.randn(2,4,9,11)
    expected_projection=channels!=4 or stride!=1
    assert detail['projection']==expected_projection
    branch=m.drop_path(m.dropout(m.attention(m.main(x))))
    assert branch.shape==m.shortcut(x).shape
    assert torch.equal(m(x),m.shortcut(x)+m.gate*branch)
    assert list(m(x).shape)==[2,*detail['output_shape'][1:]]
    assert sum(t.numel() for t in m.parameters())==detail['parameter_count']

@pytest.mark.parametrize('name',sorted(advanced.RESIDUALS))
@pytest.mark.parametrize('attention',['eca','spatial_attention','channel_attention'])
def test_internal_attention_cost_and_shape(name,attention):
    b=ModelBuilder((4,5,7),3).add(name,**dict(SAMPLES[name],attention=attention))
    m=b.build();report=ModelBuilder.preflight(b.to_config())
    assert sum(p.numel() for p in m.features[0].parameters())==report['layers'][0]['parameter_count']
    assert m(torch.randn(2,4,5,7)).shape==(2,3)

@pytest.mark.parametrize('name',sorted(SAMPLES))
def test_invalid_configuration_for_every_component(name):
    p=dict(SAMPLES[name])
    if name in advanced.RESIDUALS:p['kernel_size']=2
    elif name in advanced.ATTENTION:p['channels']=0
    elif name=='gaussian_noise':p['std']=-1
    else:p['p']=1.1
    c=ModelBuilder((4,5,7),3).to_config();c['blocks']=[dict(name=name,params=p)]
    r=ModelBuilder.preflight(c)
    assert not r['valid'] and r['diagnostics'][0]['layer_index']==0
    assert r['diagnostics'][0]['suggestion'] and r['diagnostics'][0]['message_tr']
    with pytest.raises(ModelConfigError):ModelBuilder.from_config(c).build()

def test_head_divisors_and_attention_resource_guards(monkeypatch):
    with pytest.raises(ModelConfigError):
        ModelRegistry().validate('multihead_spatial_attention',dict(channels=60,num_heads=8))
    with pytest.raises(ModelConfigError) as e:
        ModelRegistry().validate('multihead_spatial_attention',dict(channels=60,embedding_dim=60,num_heads=8))
    assert e.value.expected==[1,2,3,4,5,6,10,12,15,20,30,60]
    c=ModelBuilder((4,64,64),3).to_config()
    c['blocks']=[dict(name='spatial_self_attention',params=dict(channels=4))]
    r=ModelBuilder.preflight(c)
    assert r['diagnostics'][0]['code']=='ATTENTION_RESOURCE_LIMIT'
    assert r['diagnostics'][0]['actual']['tokens']==4096
    b=ModelBuilder((4,16,16),3).add('multihead_spatial_attention',channels=4,embedding_dim=8,num_heads=2)
    assert ModelBuilder.preflight(b.to_config(),batch_size=2)['layers'][0]['attention_cost']['matrix_elements']==262144
    assert not ModelBuilder.preflight(b.to_config(),batch_size=33)['valid']
    m=b.build().features[0]
    def forbidden(*a,**kw):pytest.fail('oversized attention reached QKV projection')
    monkeypatch.setattr(m.qkv,'forward',forbidden)
    with pytest.raises(ModelConfigError):m(torch.zeros(1,4,64,64))
    with pytest.raises(ModelConfigError):m(torch.zeros(33,4,16,16))

def test_self_attention_matches_explicit_token_algebra():
    m=ModelRegistry().create('spatial_self_attention',channels=4,embedding_dim=8).double().eval()
    x=torch.randn(2,4,3,5,dtype=torch.float64)
    q,k,v=m.qkv(x).chunk(3,dim=1)
    q,k,v=(t.flatten(2).transpose(1,2) for t in (q,k,v))
    weights=(q@k.transpose(1,2)/8**.5).softmax(-1)
    expected=m.projection((weights@v).transpose(1,2).reshape(2,8,3,5))
    assert torch.equal(m(x),expected)

@pytest.mark.parametrize('name',['dropout','dropout2d','drop_path','gaussian_noise'])
def test_seed_identity_and_no_inplace(name):
    m=ModelRegistry().create(name,**SAMPLES[name]);x=torch.ones(16,4,5,7);before=x.clone()
    torch.manual_seed(11);first=m(x)
    torch.manual_seed(11);assert torch.equal(first,m(x))
    assert not torch.equal(first,x) and torch.equal(x,before)
    m.eval();assert m(x) is x
    if name=='drop_path':assert set(first.unique().tolist())=={0.,1.25}
    if name=='dropout2d':assert all(t.unique().numel()==1 for t in first.flatten(0,1))
    if name=='gaussian_noise':
        m.train();g1=torch.Generator().manual_seed(3);g2=torch.Generator().manual_seed(3)
        assert torch.equal(m(x,generator=g1),m(x,generator=g2))

def test_gate_initialization_preactivation_and_invalid_groups():
    m=ModelRegistry().create('gated_residual',in_channels=4,out_channels=4,intermediate_channels=8,gate_init=0.,learnable_gate=False)
    x=torch.randn(2,4,5,7);assert torch.equal(m(x),x)
    r=ModelBuilder.preflight(ModelBuilder((4,5,7),3).add('gated_residual',in_channels=4,out_channels=4,intermediate_channels=8,gate_init=0.,learnable_gate=False).to_config())
    assert not r['layers'][0]['residual_active']
    p=ModelRegistry().create('preactivation_residual',**SAMPLES['preactivation_residual'])
    assert isinstance(p.main[0],nn.GroupNorm) and isinstance(p.main[1],nn.SiLU)
    with pytest.raises(ModelConfigError):ModelRegistry().create('grouped_bottleneck',in_channels=4,out_channels=4,intermediate_channels=8,groups=3)
    with pytest.raises(ModelConfigError):ModelRegistry().create('inverted_bottleneck',in_channels=8192,out_channels=4,expansion_ratio=2)

def example():
    return (ModelBuilder((3,64,64),3).add('conv2d',in_channels=3,out_channels=32)
        .add('batch_norm',num_features=32).add('silu')
        .add('bottleneck_residual',in_channels=32,out_channels=64,intermediate_channels=16)
        .add('eca',channels=64).add('max_pool2d',kernel_size=2)
        .add('inverted_bottleneck',in_channels=64,out_channels=64,expansion_ratio=2)
        .add('drop_path',p=.1).add('channel_attention',channels=64).add('global_avg_pool2d'))

def test_training_session_and_cli(tmp_path,capsys):
    b=example();model=b.build()
    config=TrainingConfig(epochs=1,batch_size=2,optimizer='sgd')
    loader,_=config.make_loaders(TensorDataset(torch.randn(2,3,64,64),torch.tensor([0,1])))
    assert config.prepare(model,loader).fit().to_dict()['applied_updates']==1
    path=tmp_path/'advanced.json';b.to_json(path)
    for action in ('validate','summary','inspect'):
        args=['models',action,str(path),'--json']+(['--execute'] if action=='inspect' else [])
        assert main(args)==0
        assert json.loads(capsys.readouterr().out)['exit_code']==0
    for name in SAMPLES:
        assert main(['models','show',name,'--json'])==0
        assert json.loads(capsys.readouterr().out)['result']['description_tr']


@pytest.mark.parametrize('name',sorted(advanced.SELF_ATTENTION))
def test_attention_dropout_seed_and_eval(name):
    p=dict(SAMPLES[name],dropout=.4)
    m=ModelRegistry().create(name,**p)
    x=torch.randn(2,4,3,5)
    torch.manual_seed(7);first=m(x)
    torch.manual_seed(7);assert torch.equal(first,m(x))
    m.eval();assert torch.equal(m(x),m(x))
    assert not torch.equal(first,m(x))


@pytest.mark.parametrize('name',['dropout','dropout2d'])
def test_native_dropout_probability_endpoints(name):
    x=torch.ones(2,4,3,5,requires_grad=True)
    assert ModelRegistry().create(name,p=0)(x) is x
    out=ModelRegistry().create(name,p=1)(x)
    assert torch.equal(out,torch.zeros_like(x))
    out.sum().backward();assert torch.isfinite(x.grad).all()


def test_extra_budget_statistics_and_token_boundary():
    b=ModelBuilder((4,32,32),3).add('spatial_self_attention',channels=4)
    assert ModelBuilder.preflight(b.to_config())['layers'][0]['attention_cost']['tokens']==1024
    c=ModelBuilder((4,3,5),3).to_config()
    c['blocks']=[dict(name='bottleneck_residual',params=dict(in_channels=4,out_channels=4,intermediate_channels=8192))]
    assert ModelBuilder.preflight(c)['diagnostics'][0]['code']=='PARAMETER_BUDGET_EXCEEDED'
    c['blocks']=[dict(name='spatial_self_attention',params=dict(channels=4,embedding_dim=3000))]
    assert ModelBuilder.preflight(c)['diagnostics'][0]['code']=='INTERMEDIATE_SHAPE_ERROR'
    b=ModelBuilder((4,1,1),3).add('bottleneck_residual',in_channels=4,out_channels=4,intermediate_channels=4)
    assert ModelBuilder.preflight(b.to_config())['warnings']
    with pytest.raises(ModelConfigError) as err:b.build()(torch.ones(1,4,1,1))
    assert err.value.layer_index==0 and err.value.code=='STATISTICS_REQUIREMENT'
