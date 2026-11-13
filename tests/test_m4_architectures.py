"""Mobile-family topology, safe metadata, CPU learning and deployment integration."""
import copy
import gc
import io
import json
import os
from pathlib import Path

import pytest

# Historical families remain available only in the development checkout profile.
pytestmark = pytest.mark.development_profile
import torch
from torch import nn

from hyso import ModelBuilder, ModelRegistry, TrainingConfig
from hyso.models import (MobileNetV2, MobileNetV3, EfficientNetV2, GhostNet, ShuffleNetV2,
                         RepVGG, MobileOne, reparameterize_model)
from hyso.models.m4_config import validate_mobile

FAMILIES = ('mobilenet_v2','mobilenet_v3','efficientnet_v2','ghostnet',
            'shufflenet_v2','repvgg','mobileone')
CANONICAL = [('mobilenet_v2',{}),('ghostnet',{})]
CANONICAL += [('mobilenet_v3',{'variant':v}) for v in ('small','large')]
CANONICAL += [('efficientnet_v2',{'variant':v}) for v in ('s','m','l')]
CANONICAL += [('shufflenet_v2',{'variant':v}) for v in ('x0_5','x1_0','x1_5','x2_0')]
CANONICAL += [('repvgg',{'variant':v}) for v in ('a0','a1')]
CANONICAL += [('mobileone',{'variant':v}) for v in ('s0','s1','s2')]


def tiny(name):
    options = dict(input_channels=1,num_classes=3)
    if name != 'shufflenet_v2':
        options['width_mult'] = .125
    else:
        options['variant'] = 'x0_5'
    if name in ('efficientnet_v2','repvgg','mobileone'):
        options['depth_mult'] = .1
    if name in ('mobilenet_v2','mobilenet_v3','efficientnet_v2','ghostnet'):
        options['dropout'] = 0.
    if name in ('efficientnet_v2',):
        options['stochastic_depth'] = 0.
    return options


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    def forbidden(*args, **kwargs):
        pytest.fail('M4 CPU test attempted CUDA initialization or work')
    for name in ('_lazy_init','current_device','synchronize'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    monkeypatch.setattr(torch.hub,'load_state_dict_from_url',lambda *a,**k:pytest.fail('weight download attempted'))


@pytest.mark.parametrize('name,options',CANONICAL)
def test_canonical_variants_create_and_forward_with_conservative_count(name,options):
    builder = ModelBuilder.named(name,input_channels=1,num_classes=3,**options)
    model = builder.build().eval()
    with torch.inference_mode():
        output = model(torch.randn(1,1,32,32))
    assert output.shape == (1,3) and torch.isfinite(output).all() and output.device.type == 'cpu'
    estimate = validate_mobile(name,builder.to_config()['params'])
    assert estimate >= sum(p.numel() for p in model.parameters())
    del model
    gc.collect()


@pytest.mark.parametrize('name',FAMILIES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_small_models_gradient_builder_json_and_weights(name,dtype,tmp_path):
    options = tiny(name)
    builder = ModelBuilder.named(name,**options)
    path = tmp_path/'architecture.json'
    builder.to_json(path)
    restored = ModelBuilder.from_json(path)
    assert restored.to_config() == builder.to_config()
    model = builder.build().to(dtype)
    assert model.to_config() == builder.to_config()
    x = torch.randn(2,1,32,32,dtype=dtype,requires_grad=True)
    output = model(x)
    assert output.shape == (2,3) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    clone = restored.build().to(dtype)
    stream = io.BytesIO()
    torch.save(model.state_dict(),stream)
    stream.seek(0)
    clone.load_state_dict(torch.load(stream,weights_only=True),strict=True)
    model.eval()
    clone.eval()
    with torch.no_grad():
        assert torch.equal(model(x),clone(x))


@pytest.mark.parametrize('name',FAMILIES)
def test_real_short_training_session_and_resume_boundary(name,tmp_path):
    from torch.utils.data import DataLoader,TensorDataset
    model = ModelBuilder.named(name,**tiny(name)).build()
    loader = DataLoader(TensorDataset(torch.randn(2,1,32,32),torch.tensor([0,1])),batch_size=2)
    config = TrainingConfig(epochs=1,batch_size=2,optimizer='sgd',learning_rate=.01)
    before = list(model.parameters())[-1].detach().clone()
    session = config.prepare(model,loader)
    result = session.fit()
    assert result.to_dict()['applied_updates'] == 1 and session.settings()['state'] == 'completed'
    assert not torch.equal(before,list(model.parameters())[-1])
    with pytest.raises(ValueError):
        TrainingConfig(epochs=1,batch_size=2,save_last_to=str(tmp_path/'last.pth')).prepare(model,loader)
    assert not (tmp_path/'last.pth').exists()


@pytest.mark.parametrize('family',['repvgg','mobileone'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_model_deploy_json_state_checkpoint_and_idempotence(family,dtype,tmp_path):
    model = ModelBuilder.named(family,**tiny(family)).build().to(dtype).eval()
    x = torch.randn(2,1,32,32,dtype=dtype)
    with torch.no_grad():
        expected = model(x)
        deployed = reparameterize_model(model)
        actual = deployed(x)
    tolerance = dict(atol=2e-6,rtol=2e-5) if dtype == torch.float32 else dict(atol=3e-12,rtol=3e-12)
    torch.testing.assert_close(actual,expected,**tolerance)
    assert not model.deploy and deployed.deploy
    config = deployed.to_config()
    assert config['params']['deploy']
    fresh = ModelBuilder.from_config(config).build().to(dtype).eval()
    path = tmp_path/'deployed-state.pth'
    torch.save(deployed.state_dict(),path)
    fresh.load_state_dict(torch.load(path,weights_only=True),strict=True)
    assert torch.equal(fresh(x),actual)
    before = {key:value.clone() for key,value in deployed.state_dict().items()}
    deployed.switch_to_deploy()
    assert all(torch.equal(value,deployed.state_dict()[key]) for key,value in before.items())
    training_state = {key:value.clone() for key,value in model.state_dict().items()}
    for strict in (True,False):
        with pytest.raises(RuntimeError,match='topology'):
            model.load_state_dict(deployed.state_dict(),strict=strict)
        assert all(torch.equal(value,model.state_dict()[key]) for key,value in training_state.items())
    directory = os.environ.get('HYSO_M4_MODEL_METRICS')
    if directory:
        record = dict(family=family,dtype=str(dtype),max_abs=(expected-actual).abs().max().item(),**tolerance)
        Path(directory,family+'-'+str(dtype).split('.')[-1]+'.json').write_text(json.dumps(record),encoding='utf-8')


def test_torchvision_canonical_state_and_eval_equivalence():
    from torchvision.models import mobilenet_v2,mobilenet_v3_small,efficientnet_v2_s,shufflenet_v2_x0_5
    pairs = [(MobileNetV2,mobilenet_v2,{}),(MobileNetV3,mobilenet_v3_small,{'variant':'small'}),
             (EfficientNetV2,efficientnet_v2_s,{}),(ShuffleNetV2,shufflenet_v2_x0_5,{'variant':'x0_5'})]
    for constructor,reference,options in pairs:
        model = constructor(num_classes=3,**options).eval()
        original = reference(weights=None,num_classes=3).eval()
        assert [(k,v.shape) for k,v in model.state_dict().items()] == [(k,v.shape) for k,v in original.state_dict().items()]
        model.load_state_dict(original.state_dict(),strict=True)
        x = torch.randn(1,3,32,32)
        with torch.no_grad():
            assert torch.equal(model(x),original(x))
        del model,original
        gc.collect()


def test_canonical_stage_definitions_and_distinct_v3_variants():
    small,large = MobileNetV3('small'),MobileNetV3('large')
    assert len(small.features) == 13 and len(large.features) == 17
    del small,large
    rep = RepVGG()
    assert rep.stage_widths == (48,96,192,1280) and rep.stage_depths == (2,4,14,1)
    one = MobileOne()
    assert one.stage_widths == (48,128,256,1024) and one.stage_depths == (2,8,10,1)
    assert len(one.stem.branches) == 1 and len(one.stages[0][0].branches) == 4
    assert len(GhostNet().blocks) == 16


@pytest.mark.parametrize('name',FAMILIES)
def test_cli_discovery_validation_summary_and_explicit_inspect(name,tmp_path,capsys,monkeypatch):
    from hyso.cli import main
    builder = ModelBuilder.named(name,**tiny(name))
    path = tmp_path/'model.json'
    builder.to_json(path)
    constructor = ModelRegistry().get(name).constructor
    original = constructor.__init__
    monkeypatch.setattr(constructor,'__init__',lambda *a,**k:pytest.fail('metadata allocated a model'))
    for args in [['models','list'],['models','show',name],['models','validate',str(path)],['models','summary',str(path)]]:
        assert main([*args,'--json']) == 0
        assert json.loads(capsys.readouterr().out)['exit_code'] == 0
    assert main(['models','inspect',str(path),'--json']) == 3
    capsys.readouterr()
    monkeypatch.setattr(constructor,'__init__',original)
    assert main(['models','inspect',str(path),'--execute','--json']) == 0
    result = json.loads(capsys.readouterr().out)['result']
    assert result['model_constructed'] and not result['forward_executed'] and result['parameters'] > 0


@pytest.mark.parametrize('name,options',[
    ('mobilenet_v2',{'width_mult':0}),('mobilenet_v3',{'variant':'medium'}),
    ('efficientnet_v2',{'variant':'b0'}),('efficientnet_v2',{'depth_mult':0}),
    ('ghostnet',{'width_mult':float('nan')}),('shufflenet_v2',{'variant':'x3_0'}),
    ('repvgg',{'variant':'b4'}),('mobileone',{'variant':'s4'}),
    ('repvgg',{'deploy':'yes'}),('mobileone',{'input_channels':True}),
    ('efficientnet_v2',{'stochastic_depth':None}),('mobilenet_v2',{'dropout':None}),
])
def test_invalid_family_parameters_allocate_nothing(name,options,monkeypatch):
    constructor = ModelRegistry().get(name).constructor
    monkeypatch.setattr(constructor,'__init__',lambda *a,**k:pytest.fail('invalid model allocated'))
    with pytest.raises(ValueError):
        ModelBuilder.named(name,**options)


def test_json_whitelist_and_parameter_budget():
    config = ModelBuilder.named('mobileone').to_config()
    config['name'] = 'os.system'
    with pytest.raises(ValueError):
        ModelBuilder.from_config(config)
    config = ModelBuilder.named('mobilenet_v2').to_config()
    config['params']['weights'] = 'DEFAULT'
    with pytest.raises(ValueError):
        ModelBuilder.from_config(config)
    with pytest.raises(ValueError,match='budget'):
        ModelBuilder.named('efficientnet_v2',variant='l',width_mult=2.,input_channels=8192,num_classes=8192)


def test_license_excluded_sources_are_unavailable():
    import hyso.models as models
    from hyso.models.m4_config import M4_FAMILIES
    from hyso.models.m4_registry import M4_BLOCKS
    assert len(ModelRegistry().list()) == 62
    assert len(M4_FAMILIES) == 7 and len(M4_BLOCKS) == 10
    assert not hasattr(models, 'FasterNet') and not hasattr(models, 'PartialConv')
    for name in ('fasternet', 'partial_conv'):
        with pytest.raises(ValueError):
            ModelRegistry().get(name)
    with pytest.raises(ValueError):
        ModelBuilder.named('fasternet')
