"""Bounded CPU parity, versioned head contracts and restricted tensor utilities."""
import copy
import json
import itertools
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from hyso import ModelBuilder,ModelRegistry,ModelConfigError,TrainingConfig,TrainingConfigError
from hyso.cli import main
from hyso.models import sampling

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    torch.manual_seed(1354)
    def forbidden(*a,**kw):pytest.fail('attempted CUDA work/query')
    for name in ('_lazy_init','is_available','device_count','synchronize'):
        monkeypatch.setattr(torch.cuda,name,forbidden)


def test_registry_counts_metadata_and_previous_records():
    r=ModelRegistry();assert len(r.list())==85 and len(sampling.NEW_NAMES)==9
    for name in sampling.NEW_NAMES:
        data=r.get(name).to_dict()
        assert data['shape_contract'] and data['cost_contract'] and data['train_eval']
        assert json.loads(json.dumps(data))==data


@pytest.mark.parametrize('mode,target,dtype',list(itertools.product(
    ('nearest','bilinear','bicubic','area'),('size','scale'),(torch.float32,torch.float64))))
@pytest.mark.parametrize('spatial',[(3,5),(8,10)])
def test_interpolation_numeric_shape_and_gradient_parity(mode,target,dtype,spatial):
    params=dict(mode=mode,**({'output_size':[7,4]} if target=='size' else {'scale_factor':[1.7,.6]}))
    if mode in ('bilinear','bicubic'):params['align_corners']=True
    b=ModelBuilder((4,*spatial),3).add('resize',**params)
    report=ModelBuilder.preflight(b.to_config())
    module=b.build().features[0]
    x=torch.randn(2,4,*spatial,dtype=dtype,requires_grad=True)
    other=x.detach().clone().requires_grad_()
    expected=F.interpolate(other,size=params.get('output_size'),scale_factor=params.get('scale_factor'),
                           mode=mode,align_corners=params.get('align_corners'))
    actual=module(x)
    assert torch.equal(actual,expected)
    assert list(actual.shape)==[2,*report['feature_shape'][1:]]
    actual.square().sum().backward();expected.square().sum().backward()
    assert torch.equal(x.grad,other.grad) and torch.isfinite(x.grad).all()
    module.eval();assert torch.equal(module(x),expected)


@pytest.mark.parametrize('mode',['bilinear','bicubic'])
@pytest.mark.parametrize('align',[None,False,True])
def test_interpolation_align_corner_variants(mode,align):
    x=torch.randn(2,4,3,5)
    m=ModelRegistry().create('resize',output_size=[7,9],mode=mode,align_corners=align)
    assert torch.equal(m(x),F.interpolate(x,size=[7,9],mode=mode,align_corners=align))


@pytest.mark.parametrize('name,params,code',[
    ('resize',{},'INVALID_PARAMETER'),('resize',dict(output_size=4,scale_factor=2),'INVALID_PARAMETER'),
    ('resize',dict(output_size=[0,4]),'INVALID_PARAMETER'),('resize',dict(scale_factor=0),'INVALID_PARAMETER'),
    ('resize',dict(scale_factor=[1,float('inf')]),'INVALID_PARAMETER'),
    ('resize',dict(scale_factor=True),'INVALID_PARAMETER'),('resize',dict(scale_factor=[1]),'INVALID_PARAMETER'),
    ('resize',dict(output_size=4,mode='linear'),'INVALID_PARAMETER'),
    ('resize',dict(output_size=4,align_corners=False),'INVALID_PARAMETER'),
    ('resize',dict(output_size=4,mode='area',align_corners=True),'INVALID_PARAMETER'),
    ('resize',dict(output_size=4,recompute_scale_factor=True),'INVALID_PARAMETER'),
    ('resize',dict(output_size=4,antialias=True),'INVALID_PARAMETER'),
    ('resize',dict(scale_factor=.01),'SPATIAL_DIMENSION_ERROR'),
    ('resize',dict(scale_factor=8192),'SPATIAL_DIMENSION_ERROR'),
    ('pixel_shuffle',dict(upscale_factor=3),'CHANNEL_MISMATCH'),
    ('pixel_unshuffle',dict(downscale_factor=2),'SPATIAL_DIMENSION_ERROR'),
    ('permute',dict(dims=[0,2,3,1]),'INVALID_PARAMETER'),
    ('permute',dict(dims=[1,0,2,3]),'INVALID_PARAMETER'),
    ('permute',dict(dims=[0,1,1,3]),'INVALID_PARAMETER'),
    ('reshape',dict(shape=[2,3,4]),'INVALID_INPUT_SHAPE'),
    ('reshape',dict(shape=[-1,3,5]),'INVALID_PARAMETER'),
    ('reshape',dict(shape=[1,8193,1]),'INVALID_PARAMETER'),
    ('flatten',{},'INVALID_BLOCK_ORDER'),
    ('regression_head',dict(in_features=4,output_features=2),'INVALID_BLOCK_ORDER'),
])
def test_invalid_layer_diagnostics(name,params,code):
    c=ModelBuilder((4,3,5),3).to_config();c['blocks']=[dict(name=name,params=params)]
    report=ModelBuilder.preflight(c)
    assert not report['valid'] and report['diagnostics'][0]['code']==code
    error=report['diagnostics'][0]
    expected_index = None if params.get('scale_factor')==[1,float('inf')] else 0
    assert error['layer_index']==expected_index and error['message_tr'] and error['suggestion']
    with pytest.raises(ModelConfigError):ModelBuilder.from_config(c).build()


@pytest.mark.parametrize('factor',[1,2,3])
def test_pixel_rearrangement_roundtrip_and_gradients(factor):
    b=ModelBuilder((4*factor*factor,3,5),3).add('pixel_shuffle',upscale_factor=factor)
    m=b.build().features[0];x=torch.randn(2,4*factor*factor,3,5,requires_grad=True)
    output=m(x)
    assert torch.equal(output,nn.PixelShuffle(factor)(x))
    reverse=ModelRegistry().create('pixel_unshuffle',downscale_factor=factor)
    assert torch.equal(reverse(output),x)
    output.sum().backward();assert torch.isfinite(x.grad).all()
    assert list(output.shape)==[2,*ModelBuilder.preflight(b.to_config())['feature_shape'][1:]]
    reverse_b=ModelBuilder(tuple(output.shape[1:]),3).add('pixel_unshuffle',downscale_factor=factor)
    assert ModelBuilder.preflight(reverse_b.to_config())['feature_shape']==['N',4*factor*factor,3,5]


def test_utility_values_axes_and_explicit_flatten_boundary():
    x=torch.arange(120.).reshape(2,4,3,5)
    b=ModelBuilder((4,3,5),3).add('permute').add('reshape',shape=[2,5,6])
    assert torch.equal(b.build().features(x),x.permute(0,1,3,2).reshape(2,2,5,6))
    b=ModelBuilder.sequential((4,3,5),head='regression',output_features=2,pooling='none').add('flatten')
    report=ModelBuilder.preflight(b.to_config())
    assert report['feature_shape']==['N',60] and report['head']['in_features']==60
    assert torch.equal(b.build().features(x),x.flatten(1))
    for name,params in [('conv2d',dict(in_channels=60,out_channels=4)),('identity',{}),('permute',{})]:
        with pytest.raises(ModelConfigError):b.add(name,**params)
    c=b.to_config();c['head']['params']['pooling']='avg'
    assert ModelBuilder.preflight(c)['diagnostics'][0]['code']=='INVALID_OUTPUT_HEAD'
    with pytest.raises(ModelConfigError):ModelBuilder.sequential((8192,2,2),head='regression',output_features=1,pooling='none').add('flatten')


@pytest.mark.parametrize('head,params',[
    ('classification',dict(num_classes=3,dropout=.2)),('regression',dict(output_features=2)),
    ('embedding',dict(embedding_dim=5,normalize=True)),('embedding',dict(embedding_dim=4,normalize=False))])
@pytest.mark.parametrize('pooling',['avg','max','none'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_head_json_modes_exact_parameters_training_and_state(head,params,pooling,dtype,tmp_path):
    b=ModelBuilder.sequential((4,3,5),head=head,pooling=pooling,**params).add('silu')
    path=tmp_path/'architecture.json';b.to_json(path)
    clone_b=ModelBuilder.from_json(path);assert clone_b.to_config()==b.to_config()
    assert b.to_config()['schema_version']==2 and b.to_config()['kind']=='sequential'
    report=ModelBuilder.preflight(b.to_config())
    m=b.build().to(dtype);assert sum(p.numel() for p in m.parameters())==report['parameter_estimate']
    optimizer=torch.optim.SGD(m.parameters(),lr=.01)
    for training in (True,False):
        m.train(training);optimizer.zero_grad(set_to_none=True)
        x=torch.randn(2,4,3,5,dtype=dtype,requires_grad=True);out=m(x)
        assert list(out.shape)==[2,report['head']['output_features']] and torch.isfinite(out).all()
        target=torch.randn_like(out)
        F.mse_loss(out,target).backward();optimizer.step()
        assert torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    other=clone_b.build().to(dtype);other.load_state_dict(copy.deepcopy(m.state_dict()),strict=True)
    m.eval();other.eval();assert torch.equal(m(x),other(x))
    if head=='embedding' and params['normalize']:
        assert torch.allclose(m(x).norm(dim=1),torch.ones(2,dtype=dtype))
    if head!='classification':
        config=TrainingConfig(epochs=1,batch_size=2)
        loader,_=config.make_loaders(TensorDataset(torch.randn(2,4,3,5),torch.tensor([0,1])))
        with pytest.raises(TrainingConfigError,match='explicit loss'):config.prepare(m.float(),loader)


def test_zero_embedding_and_unrestricted_regression():
    m=ModelRegistry().create('embedding_head',in_features=4,embedding_dim=3,normalize=True,eps=1e-300)
    nn.init.zeros_(m.linear.weight);nn.init.zeros_(m.linear.bias)
    x=torch.zeros(2,4,3,5,requires_grad=True);out=m(x)
    assert torch.equal(out,torch.zeros_like(out))
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and all(torch.isfinite(p.grad).all() for p in m.parameters())
    # The zero output remains finite even with an epsilon below float32 range.
    reg=ModelRegistry().create('regression_head',in_features=4,output_features=2)
    nn.init.zeros_(reg.linear.weight);nn.init.constant_(reg.linear.bias,-5.)
    assert torch.equal(reg(torch.zeros(2,4,3,5)),torch.full((2,2),-5.))


@pytest.mark.parametrize('head',[
    {},dict(name='missing',params={}),dict(name='regression_head',params=dict(output_features=0)),
    dict(name='embedding_head',params=dict(embedding_dim=3,normalize='yes')),
    dict(name='classification_head',params=dict(num_classes=True)),
    dict(name='regression_head',params=dict(output_features=3,in_features=4)),
])
def test_invalid_schema2_heads(head):
    c=dict(format='hyso.model_config',schema_version=2,kind='sequential',input_shape=[4,3,5],blocks=[],head=head)
    assert ModelBuilder.preflight(c)['diagnostics'][0]['code']=='INVALID_OUTPUT_HEAD'


def test_preflight_fences_budget_and_schema1_exact_compatibility(monkeypatch):
    def forbidden(*a,**kw):pytest.fail('preflight allocated a module/tensor')
    old=ModelBuilder((4,3,5),3).add('conv2d',in_channels=4,out_channels=8)
    torch.manual_seed(4);model=old.build().eval()
    torch.manual_seed(4);reference=nn.Sequential(nn.Conv2d(4,8,3,padding=1),nn.AdaptiveAvgPool2d(1),nn.Flatten(1),nn.Linear(8,3)).eval()
    x=torch.randn(2,4,3,5);assert torch.equal(model(x),reference(x))
    assert list(model.state_dict())==['features.0.weight','features.0.bias','classifier.weight','classifier.bias']
    assert ModelBuilder.from_config(old.to_config()).to_config()['schema_version']==1
    monkeypatch.setattr(nn.Module,'__init__',forbidden);monkeypatch.setattr(torch,'empty',forbidden)
    b=ModelBuilder.sequential((4,3,5),head='embedding',embedding_dim=3).add('resize',scale_factor=2)
    assert ModelBuilder.preflight(b.to_config())['head']['parameter_count']==15
    assert ModelBuilder.preflight(b.to_config(),activation_limit=100)['warnings']
    c=b.to_config();c['head']['params']['embedding_dim']=8192;c['input_shape'][0]=8192
    assert ModelBuilder.preflight(c)['diagnostics'][0]['code']=='PARAMETER_BUDGET_EXCEEDED'


def test_schema2_classification_session_and_cli(tmp_path,capsys):
    b=ModelBuilder.sequential((4,3,5),head='classification',num_classes=3).add('resize',output_size=[4,6])
    config=TrainingConfig(epochs=1,batch_size=2,optimizer='sgd')
    loader,_=config.make_loaders(TensorDataset(torch.randn(2,4,3,5),torch.tensor([0,1])))
    assert config.prepare(b.build(),loader).fit().to_dict()['applied_updates']==1
    path=tmp_path/'model.json';b.to_json(path)
    for action in ('validate','summary','inspect'):
        args=['models',action,str(path),'--json']+(['--execute'] if action=='inspect' else [])
        assert main(args)==0
        value=json.loads(capsys.readouterr().out);assert value['exit_code']==0
        if action=='summary':assert value['result']['summary']['head']['name']=='classification_head'
    c=b.to_config();c['schema_version']=1
    path.write_text(json.dumps(c));assert main(['models','validate',str(path),'--json'])==2
    assert json.loads(capsys.readouterr().out)['diagnostics']
    path.write_text('{"format":"x","format":"y"}')
    assert main(['models','validate',str(path),'--json'])==2
    capsys.readouterr()


def test_head_activation_warning_and_standalone_feature_validation():
    b=ModelBuilder.sequential((1,2,2),head='regression',output_features=200)
    r=ModelBuilder.preflight(b.to_config(),activation_limit=100)
    assert r['valid'] and r['head']['warnings'][0]['layer_id']=='head'
    assert r['activation_estimate']['peak_tensor_elements']==200
    m=ModelRegistry().create('regression_head',in_features=4,output_features=2,pooling='none')
    for shape in ((2,3),(2,2,2),(0,4)):
        with pytest.raises(ModelConfigError):m(torch.ones(shape))
