"""CPU registry, architecture, JSON security and external-model compatibility."""
from dataclasses import FrozenInstanceError
import json

import pytest
import torch
from torch import nn

from hyso import ModelBuilder, ModelRegistry, ModelConfigError, ModularCNN, Trainer, BenchmarkRunner
from hyso.models.blocks import ConvNormAct, ResidualBlock


def builder():
    return ModelBuilder((1,8,8),2).add('conv_norm_act',in_channels=1,out_channels=8,norm=None)


def test_catalog_is_real_immutable_and_plain_metadata():
    r = ModelRegistry()
    assert len(r.list()) == 62
    assert r.get('conv_norm_act').constructor is ConvNormAct
    assert r.get('residual').constructor is ResidualBlock
    assert r.get('modular_cnn').constructor is ModularCNN
    assert [s.name for s in r.list('attention')] == ['se','cbam','eca','coordinate','simam','triplet','selective_kernel']
    for spec in r.list():
        assert spec.input_condition and spec.output_condition and spec.compatibility
        assert json.loads(json.dumps(spec.to_dict()))['name'] == spec.name
        with pytest.raises(FrozenInstanceError): spec.name = 'changed'
    for name in ('mobile_net','__import__','os.system'):
        with pytest.raises(ModelConfigError): r.get(name)
    with pytest.raises(ModelConfigError): r.list('unknown')


@pytest.mark.parametrize('name,params', [
    ('conv2d',dict(in_channels=8,out_channels=8,groups=8)),
    ('conv_norm_act',dict(in_channels=8,out_channels=8,groups=8,norm='group')),
    ('residual',dict(in_channels=8,out_channels=16,stride=2,
                     attention=dict(name='cbam',params=dict(channels=16)))),
    ('se',dict(channels=8)), ('cbam',dict(channels=8,reduction=4)),
    ('eca',dict(channels=8)), ('coordinate',dict(channels=8)),
    ('batch_norm',dict(num_features=8)), ('group_norm',dict(num_groups=2,num_channels=8)),
    ('relu',{}), ('silu',{}), ('hardswish',{}), ('identity',{})])
def test_each_block_forward_backward_matches_declared_shape(name,params):
    x = torch.randn(2,8,8,8,requires_grad=True)
    b = ModelBuilder((8,8,8),3).add(name,**params)
    m = b.build()
    assert m.features[0].__class__ is ModelRegistry().get(name).constructor
    out = m(x)
    assert out.shape == (2,3)
    out.sum().backward()
    assert x.grad is not None and all(p.device.type == 'cpu' for p in m.parameters())


def test_chains_detach_config_and_json_roundtrip(tmp_path):
    b = builder().add('residual',in_channels=8,out_channels=8,norm=None,
                      attention=dict(name='eca',params=dict(channels=8))).add('silu')
    path = tmp_path/'architecture.json'
    b.to_json(path)
    with pytest.raises(FileExistsError): b.to_json(path)
    clone = ModelBuilder.from_json(path)
    assert clone.to_config() == b.to_config()
    m, other = b.build().eval(), clone.build().eval()
    other.load_state_dict(m.state_dict())
    x = torch.randn(2,1,8,8)
    assert torch.equal(m(x),other(x))
    c = b.to_config(); c['blocks'][0]['params']['out_channels'] = 123
    assert b.to_config()['blocks'][0]['params']['out_channels'] == 8
    assert m.to_config() == b.to_config()


def test_modular_profile_is_original_class_weights_and_checkpoint_layout(tmp_path):
    p = dict(input_channels=1,num_classes=2,widths=(8,),repeats=(1,),strides=(1,),attention='eca',norm=None)
    torch.manual_seed(5); original = ModularCNN(**p).eval()
    b = ModelBuilder.modular(**p)
    b.to_json(tmp_path/'modular.json')
    torch.manual_seed(5); built = ModelBuilder.from_json(tmp_path/'modular.json').build().eval()
    assert type(built) is ModularCNN
    assert list(built.state_dict()) == list(original.state_dict())
    x = torch.randn(2,1,8,8)
    assert torch.equal(built(x),original(x))
    from hyso.checkpoint import save_training_checkpoint, load_training_checkpoint
    save_training_checkpoint(tmp_path/'state.pth',built)
    load_training_checkpoint(tmp_path/'state.pth',original)
    assert torch.equal(built(x),original(x))
    with pytest.raises(ModelConfigError): b.add('identity')


@pytest.mark.parametrize('name,params', [
    ('conv2d',dict(in_channels=2,out_channels=4)),
    ('conv2d',dict(in_channels=1,out_channels=4,kernel_size=20,padding=0)),
    ('group_norm',dict(num_channels=1,num_groups=2)),
    ('eca',dict(channels=1,kernel_size=2)),
    ('se',dict(channels=1,reduction=0)),
    ('residual',dict(in_channels=1,out_channels=8,attention=dict(name='eca',params=dict(channels=4)))),
    ('relu',dict(inplace=True)), ('modular_cnn',{}),
    ('conv2d',dict(in_channels=True,out_channels=4)),
    ('conv2d',dict(in_channels=1,out_channels=4,device='cuda'))])
def test_bad_blocks_are_rejected_transactionally(name,params):
    b = ModelBuilder((1,8,8),2)
    before = b.to_config()
    with pytest.raises(ModelConfigError): b.add(name,**params)
    assert b.to_config() == before


def test_shape_and_statistics_errors_are_explicit():
    m = builder().build()
    for x in (torch.randn(1,1,7,8),torch.randn(1,8,8),torch.randn(1,2,8,8),torch.randn(1,1,8,8).double()):
        with pytest.raises(ModelConfigError): m(x)
    bn = ModelBuilder((2,1,1),2).add('batch_norm',num_features=2).build()
    with pytest.raises(ModelConfigError,match='block 0'): bn(torch.ones(1,2,1,1))
    assert bn.eval()(torch.ones(1,2,1,1)).shape == (1,2)


@pytest.mark.parametrize('raw', [
    '{"format":"hyso.model_config","format":"evil"}',
    '{"value":NaN}', '{"value":Infinity}', '{"value":1e999}',
    '[]', '{"module":"os","constructor":"system","command":"echo bad"}',
    '{"format":"hyso.model_config","schema_version":true,"kind":"modular_cnn","params":{}}'])
def test_untrusted_json_is_data_only(tmp_path,raw):
    path=tmp_path/'bad.json'; path.write_text(raw,encoding='utf-8')
    with pytest.raises(ModelConfigError): ModelBuilder.from_json(path)


def test_invalid_graph_and_budgets_fail_before_construction(monkeypatch,tmp_path):
    def forbidden(*a,**k): pytest.fail('invalid config reached a constructor')
    monkeypatch.setattr(ConvNormAct,'__init__',forbidden)
    c = ModelBuilder((8192,8,8),2).to_config()
    c['blocks'] = [dict(name='conv_norm_act',params=dict(in_channels=8192,out_channels=8192,kernel_size=3))]
    with pytest.raises(ModelConfigError,match='parameter budget'): ModelBuilder.from_config(c).build()
    c=ModelBuilder((1,8,8),2).to_config();c['connections'] = [[0,1]]
    with pytest.raises(ModelConfigError): ModelBuilder.from_config(c)
    c=ModelBuilder((1,8,8),2).to_config();c['blocks']=[dict(name='identity',params={})]*65
    with pytest.raises(ModelConfigError): ModelBuilder.from_config(c)
    p=tmp_path/'large.json';p.write_bytes(b' '*(2*1024**2+1))
    with pytest.raises(ModelConfigError): ModelBuilder.from_json(p)


@pytest.mark.parametrize('name',['se','cbam','eca','coordinate'])
def test_small_attention_channels_retain_original_semantics(name):
    m=ModelBuilder((1,1,1),2).add(name,channels=1).build()
    out=m(torch.ones(1,1,1,1,requires_grad=True))
    assert out.shape==(1,2)
    out.sum().backward()


def test_modular_rejects_incompatible_stages_and_accepts_small_se():
    with pytest.raises(ModelConfigError): ModelBuilder.modular(widths=(8,16),repeats=(1,),strides=(1,2))
    with pytest.raises(ModelConfigError): ModelBuilder.modular(widths=(8192,),repeats=(1,),strides=(1,))
    assert type(ModelBuilder.modular(widths=(1,),repeats=(1,),strides=(1,),attention='se').build()) is ModularCNN


@pytest.mark.parametrize('external',[False,True])
def test_plain_external_model_and_built_model_use_existing_apis(external):
    from torch.utils.data import DataLoader,TensorDataset
    m = nn.Sequential(nn.Flatten(),nn.Linear(64,2)) if external else builder().build()
    x,y=torch.randn(4,1,8,8),torch.tensor([0,1,0,1])
    t=Trainer(m,optimizer=torch.optim.SGD(m.parameters(),lr=.01),loss_fn=nn.CrossEntropyLoss(),device='cpu')
    assert len(t.fit(DataLoader(TensorDataset(x,y),batch_size=2),epochs=1)['train_loss'])==1
    result=BenchmarkRunner(m,example_input=x[:1],warmup_iterations=1,measure_iterations=2,repeats=1,device='cpu').run()
    assert result.to_dict()['configuration']['measure_iterations']==2
