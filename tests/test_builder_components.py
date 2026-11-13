"""Stage 13.5.2 deterministic CPU contracts, grouped by acceptance category."""
import copy
import itertools
import json

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from hyso import ModelBuilder, ModelRegistry, ModelConfigError, TrainingConfig
from hyso.cli import main
from hyso.models import components
from hyso.models.registry import _DIAGNOSTICS


SAMPLES = {
    'pointwise_conv2d': dict(in_channels=4,out_channels=6,stride=[2,3]),
    'depthwise_conv2d': dict(in_channels=4,depth_multiplier=2,dilation=[2,3],padding=[2,3]),
    'depthwise_separable_conv2d': dict(in_channels=4,out_channels=6,depth_multiplier=2),
    'spatial_separable_conv2d': dict(in_channels=4,out_channels=6,intermediate_channels=3,kernel_size=[2,5],padding=[0,2]),
    'factorized_conv2d': dict(in_channels=4,out_channels=6,intermediate_channels=2),
    'conv_transpose2d': dict(in_channels=4,out_channels=6,groups=2,stride=[2,3],output_padding=[1,2]),
    'channel_layer_norm': dict(channels=4),
    'instance_norm': dict(num_features=4,affine=True,track_running_stats=True),
    'leaky_relu': dict(negative_slope=.2), 'prelu': dict(num_parameters=4,init=-.1),
    'gelu': dict(approximate='tanh'), 'mish': {}, 'hardsigmoid': {},
    'max_pool2d': dict(kernel_size=[2,3],stride=[2,3],ceil_mode=True),
    'avg_pool2d': dict(kernel_size=3,stride=2,padding=1,count_include_pad=False,ceil_mode=True),
    'adaptive_avg_pool2d': dict(output_size=[3,5]),
    'adaptive_max_pool2d': dict(output_size=[3,5]), 'global_avg_pool2d': {},
}


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    torch.manual_seed(1352)
    def forbidden(*args, **kwargs):
        pytest.fail('component test attempted CUDA work')
    for name in ('_lazy_init', 'is_available', 'device_count', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def test_a_registry_metadata_and_previous_names():
    assert set(SAMPLES) == components.NEW_NAMES
    registry = ModelRegistry()
    assert len(registry.list()) == 85
    old = {'conv2d','conv_norm_act','residual','se','cbam','eca','coordinate','batch_norm',
           'group_norm','relu','silu','hardswish','identity','modular_cnn','channel_shuffle','dropblock2d'}
    assert old | components.NEW_NAMES <= {s.name for s in registry.list()}
    assert len(_DIAGNOSTICS) == 18
    for name in SAMPLES:
        row = registry.get(name).to_dict()
        assert row['layout'] == 'NCHW' and row['description'] and row['common_errors']
        assert json.loads(json.dumps(row)) == row
        assert all('constraints' in p or p['kind'] == 'choice' for p in row['parameters'])


@pytest.mark.parametrize('name,params', list(SAMPLES.items()))
def test_f_preflight_allocates_no_weights(name, params, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('preflight allocated a module/tensor')
    monkeypatch.setattr(nn.Module, '__init__', forbidden)
    monkeypatch.setattr(torch, 'empty', forbidden)
    builder = ModelBuilder((4,13,17),3).add(name,**params)
    report = ModelBuilder.preflight(builder.to_config(), batch_size=2)
    assert report['valid'] and not report['diagnostics']
    layer = report['layers'][0]
    assert layer['parameter_count'] is not None
    assert layer['parameter_estimate'] == layer['parameter_count']
    assert layer['parameter_estimate_kind'] == 'conservative_upper_bound'
    assert report['activation_estimate']['batch_size'] == 2


@pytest.mark.parametrize('name,params', list(SAMPLES.items()))
def test_g_h_i_json_cpu_gradients_optimizer_and_strict_state(name, params, tmp_path):
    builder = ModelBuilder((4,13,17),3).add(name,**params)
    path = tmp_path/'architecture.json'
    builder.to_json(path)
    restored = ModelBuilder.from_json(path)
    assert restored.to_config() == builder.to_config()
    assert restored.to_config()['schema_version'] == 1
    model = builder.build()
    preflight = ModelBuilder.preflight(builder.to_config())
    assert sum(p.numel() for p in model.features[0].parameters()) == preflight['layers'][0]['parameter_count']
    optimizer = torch.optim.SGD(model.parameters(),lr=.01)
    for training in (True, False):
        model.train(training)
        optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(2,4,13,17,requires_grad=True)
        features = model.features(inputs)
        assert list(features.shape) == [2,*preflight['feature_shape'][1:]]
        output = model(inputs)
        assert output.shape == (2,3) and torch.isfinite(output).all()
        output.square().sum().backward()
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    clone = restored.build()
    clone.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    model.eval(); clone.eval()
    inputs = torch.randn(2,4,13,17)
    assert torch.equal(model(inputs),clone(inputs))


@pytest.mark.parametrize('kernel,stride,dilation,groups', list(itertools.product((2,3),(1,2,3),(1,2,3),(1,2,4))))
def test_b_convolution_asymmetric_parity(kernel,stride,dilation,groups):
    params = dict(in_channels=4,out_channels=8,kernel_size=[kernel,kernel+1],
                  stride=[stride,4-stride],dilation=[dilation,4-dilation],groups=groups,padding=[2,3])
    builder = ModelBuilder((4,13,17),3).add('conv2d',**params)
    report = ModelBuilder.preflight(builder.to_config())
    module = ModelRegistry().create('conv2d',**params)
    assert list(module(torch.zeros(2,4,13,17)).shape) == [2,*report['feature_shape'][1:]]
    assert sum(p.numel() for p in module.parameters()) == report['layers'][0]['parameter_count']
    assert report['layers'][0]['parameter_estimate'] >= report['layers'][0]['parameter_count']


@pytest.mark.parametrize('stride,dilation,output_padding', [(1,2,1),(2,1,1),(3,2,2),(1,3,2)])
def test_b_transpose_output_padding_uses_or_rule(stride,dilation,output_padding):
    params = dict(in_channels=4,out_channels=4,stride=stride,dilation=dilation,output_padding=output_padding)
    builder = ModelBuilder((4,3,5),3).add('conv_transpose2d',**params)
    assert list(builder.build().features(torch.zeros(2,4,3,5)).shape) == [2,*ModelBuilder.preflight(builder.to_config())['feature_shape'][1:]]


@pytest.mark.parametrize('name', ['depthwise_separable_conv2d','spatial_separable_conv2d','factorized_conv2d'])
def test_b_multistage_shape_parity(name):
    params = dict(SAMPLES[name],stride=[2,3],dilation=[2,1],padding=[2,2])
    builder = ModelBuilder((4,13,17),3).add(name,**params)
    report = ModelBuilder.preflight(builder.to_config())['layers'][0]
    inputs = torch.zeros(2,4,13,17)
    outputs = []
    for child in builder.build().features[0].children():
        inputs = child(inputs)
        outputs.append(list(inputs.shape))
    assert outputs == [[2,*s[1:]] for s in report['intermediate_shapes']+[report['output_shape']]]


@pytest.mark.parametrize('name', sorted(components.CONVOLUTIONS))
def test_b_bias_disabled_exact_cost(name):
    params = dict(SAMPLES.get(name, dict(in_channels=4,out_channels=6)),bias=False)
    builder = ModelBuilder((4,13,17),3).add(name,**params)
    report = ModelBuilder.preflight(builder.to_config())
    model = builder.build()
    assert sum(p.numel() for p in model.features[0].parameters()) == report['layers'][0]['parameter_count']
    assert model(torch.randn(2,4,13,17)).shape == (2,3)


@pytest.mark.parametrize('name,params,reference', [
    ('batch_norm',dict(num_features=4,eps=.002,momentum=None,affine=False),nn.BatchNorm2d),
    ('group_norm',dict(num_groups=2,num_channels=4,eps=.002,affine=False),nn.GroupNorm),
    ('instance_norm',dict(num_features=4,eps=.002,momentum=.2,affine=True,track_running_stats=True),nn.InstanceNorm2d),
])
def test_c_normalization_native_parity(name,params,reference):
    module = ModelRegistry().create(name,**params)
    native = reference(**params)
    native.load_state_dict(module.state_dict(),strict=True)
    x = torch.randn(2,4,5,7)
    for training in (True,False):
        module.train(training); native.train(training)
        assert torch.equal(module(x),native(x))


def test_c_channel_layer_norm_explicit_axis_and_runtime_channels():
    module = ModelRegistry().create('channel_layer_norm',channels=4)
    x = torch.randn(2,4,3,5)
    native = nn.LayerNorm(4)
    native.load_state_dict(module.norm.state_dict())
    assert torch.equal(module(x),native(x.movedim(1,-1)).movedim(-1,1))
    for name in ('channel_layer_norm','instance_norm'):
        module = ModelRegistry().create(name,**({'channels':4} if name=='channel_layer_norm' else {'num_features':4}))
        for bad in (torch.zeros(2,3,5,7),torch.zeros(2,5,7,4)):
            with pytest.raises(ModelConfigError): module(bad)


@pytest.mark.parametrize('name,params,reference', [
    ('leaky_relu',dict(negative_slope=.2),nn.LeakyReLU), ('prelu',dict(num_parameters=4,init=-.1),nn.PReLU),
    ('gelu',dict(approximate='none'),nn.GELU), ('gelu',dict(approximate='tanh'),nn.GELU),
    ('mish',{},nn.Mish), ('hardsigmoid',{},nn.Hardsigmoid),
])
def test_d_activation_native_parity(name,params,reference):
    x = torch.linspace(-4,4,48).reshape(1,4,3,4)
    assert torch.equal(ModelRegistry().create(name,**params)(x),reference(**params)(x))


@pytest.mark.parametrize('name,ceil,kernel,stride,size', list(itertools.product(('max_pool2d','avg_pool2d'),(False,True),(2,3),(1,2,3),(1,2,5))))
def test_e_pooling_ceil_floor_boundaries(name,ceil,kernel,stride,size):
    params = dict(kernel_size=[kernel,kernel+1],stride=[stride,4-stride],padding=[kernel//2,(kernel+1)//2],ceil_mode=ceil)
    config = dict(format='hyso.model_config',schema_version=1,kind='sequential_classifier',input_shape=[4,size,size+2],num_classes=3,
                  blocks=[dict(name=name,params=params)])
    report = ModelBuilder.preflight(config)
    module = ModelRegistry().create(name,**params)
    try:
        actual = module(torch.randn(2,4,size,size+2))
    except RuntimeError:
        assert not report['valid']
    else:
        assert report['valid'],report['diagnostics']
        assert list(actual.shape) == [2,*report['feature_shape'][1:]]


@pytest.mark.parametrize('dilation', [1,2,3])
def test_e_max_pool_dilation_parity(dilation):
    builder = ModelBuilder((4,11,13),3).add('max_pool2d',kernel_size=[2,3],stride=[2,3],padding=[1,1],dilation=dilation,ceil_mode=True)
    report = ModelBuilder.preflight(builder.to_config())
    assert list(builder.build().features(torch.randn(2,4,11,13)).shape) == [2,*report['feature_shape'][1:]]


@pytest.mark.parametrize('include_pad', [True,False])
def test_e_avg_pool_divisor_semantics(include_pad):
    params = dict(kernel_size=3,stride=2,padding=1,count_include_pad=include_pad,ceil_mode=True)
    x = torch.arange(140.).reshape(1,4,5,7)
    assert torch.equal(ModelRegistry().create('avg_pool2d',**params)(x),nn.AvgPool2d(**params)(x))


@pytest.mark.parametrize('name', ['adaptive_avg_pool2d','adaptive_max_pool2d'])
@pytest.mark.parametrize('size', [1,[2,3],[9,11]])
def test_e_adaptive_output_size_parity(name,size):
    builder = ModelBuilder((4,3,5),3).add(name,output_size=size)
    report = ModelBuilder.preflight(builder.to_config())
    assert list(builder.build().features(torch.randn(2,4,3,5)).shape) == [2,*report['feature_shape'][1:]]


@pytest.mark.parametrize('name,params,code', [
    ('conv2d',dict(in_channels=4,out_channels=6,groups=4),'GROUP_DIVISIBILITY_ERROR'),
    ('conv2d',dict(in_channels=4,out_channels=4,dilation=0),'INVALID_PARAMETER'),
    ('conv2d',dict(in_channels=4,out_channels=4,dilation=[1,-1]),'INVALID_PARAMETER'),
    ('conv2d',dict(in_channels=4,out_channels=4,padding_mode='reflect'),'INVALID_PARAMETER'),
    ('conv_transpose2d',dict(in_channels=4,out_channels=4,stride=2,output_padding=2),'INVALID_OUTPUT_PADDING'),
    ('conv_transpose2d',dict(in_channels=4,out_channels=4,output_padding=-1),'INVALID_PARAMETER'),
    ('factorized_conv2d',dict(in_channels=4,out_channels=4,intermediate_channels=5),'INVALID_PARAMETER'),
    ('depthwise_conv2d',dict(in_channels=8192,depth_multiplier=2),'INTERMEDIATE_SHAPE_ERROR'),
    ('batch_norm',dict(num_features=4,eps=0),'INVALID_PARAMETER'),
    ('batch_norm',dict(num_features=4,momentum=1.1),'INVALID_PARAMETER'),
    ('group_norm',dict(num_channels=4,num_groups=3),'GROUP_DIVISIBILITY_ERROR'),
    ('channel_layer_norm',dict(channels=4,affine='yes'),'INVALID_PARAMETER'),
    ('instance_norm',dict(num_features=4,momentum=None),'INVALID_PARAMETER'),
    ('prelu',dict(num_parameters=3),'CHANNEL_MISMATCH'),
    ('prelu',dict(num_parameters=0),'INVALID_PARAMETER'),
    ('prelu',dict(init=1e100),'INVALID_PARAMETER'),
    ('gelu',dict(approximate='fast'),'INVALID_PARAMETER'),
    ('leaky_relu',dict(negative_slope=-1),'INVALID_PARAMETER'),
    ('avg_pool2d',dict(kernel_size=2,padding=2),'INVALID_POOLING_CONFIGURATION'),
    ('max_pool2d',dict(kernel_size=100),'SPATIAL_DIMENSION_ERROR'),
    ('adaptive_avg_pool2d',dict(output_size=[0,2]),'INVALID_PARAMETER'),
    ('adaptive_max_pool2d',dict(output_size=[1,8193]),'INVALID_PARAMETER'),
    ('missing_layer',{},'UNSUPPORTED_COMPONENT'),
])
def test_f_invalid_components(name,params,code):
    config = ModelBuilder((4,5,7),3).to_config()
    config['blocks'] = [dict(name=name,params=params)]
    report = ModelBuilder.preflight(config)
    assert not report['valid']
    assert report['diagnostics'][0]['code'] == code
    diagnostic = report['diagnostics'][0]
    assert set(diagnostic) == {'code','message_tr','message_en','layer_index','layer_id','expected','actual','suggestion'}
    assert diagnostic['layer_index'] == 0 and diagnostic['suggestion']
    with pytest.raises(ModelConfigError): ModelBuilder.from_config(config).build()


def test_f_intermediate_shape_cannot_be_hidden_by_later_padding():
    config = ModelBuilder((4,1,7),3).to_config()
    config['blocks'] = [dict(name='spatial_separable_conv2d',params=dict(in_channels=4,out_channels=4,
        intermediate_channels=2,kernel_size=3,padding=0))]
    assert ModelBuilder.preflight(config)['diagnostics'][0]['code'] == 'INTERMEDIATE_SHAPE_ERROR'


def test_f_statistics_warnings_and_native_runtime_failures():
    for name,params in [('batch_norm',dict(num_features=4)),('group_norm',dict(num_channels=4,num_groups=4)),
                        ('instance_norm',dict(num_features=4,track_running_stats=True))]:
        builder = ModelBuilder((4,1,1),3).add(name,**params)
        report = ModelBuilder.preflight(builder.to_config())
        assert report['valid'] and not report['diagnostics']
        assert report['warnings'][0]['code'] == 'STATISTICS_REQUIREMENT'
        with pytest.raises(ModelConfigError) as exc: builder.build()(torch.ones(1,4,1,1))
        assert exc.value.code == 'STATISTICS_REQUIREMENT'
        if name != 'group_norm': assert builder.build().eval()(torch.ones(1,4,1,1)).shape == (1,3)
    c = ModelBuilder((4,1,1),3).to_config()
    c['blocks'] = [dict(name='instance_norm',params=dict(num_features=4))]
    assert not ModelBuilder.preflight(c)['valid']
    # Batch remains symbolic for statistics even when resource estimation assumes N=2.
    builder = ModelBuilder((4,1,1),3).add('batch_norm',num_features=4,track_running_stats=False)
    report = ModelBuilder.preflight(builder.to_config(),batch_size=2)
    assert report['warnings'][0]['actual']['batch'] == 'symbolic'
    with pytest.raises(ModelConfigError): builder.build().eval()(torch.ones(1,4,1,1))


def test_f_resource_warning_and_parameter_budget():
    builder = ModelBuilder((4,5,7),3).add('depthwise_separable_conv2d',in_channels=4,out_channels=8,depth_multiplier=3)
    report = ModelBuilder.preflight(builder.to_config(),batch_size=2,activation_limit=500)
    assert report['valid'] and any(w['code']=='RESOURCE_LIMIT_WARNING' for w in report['warnings'])
    assert report['activation_estimate']['peak_tensor_elements'] == 2*12*5*7
    assert report['activation_estimate']['summed_tensor_elements'] == 2*(4+12+8)*5*7
    huge = ModelBuilder((8192,8192,8192),2).to_config()
    assert ModelBuilder.preflight(huge)['warnings'][0]['code'] == 'RESOURCE_LIMIT_WARNING'
    huge['blocks'] = [dict(name='conv_transpose2d',params=dict(in_channels=8192,out_channels=8192))]
    assert ModelBuilder.preflight(huge)['diagnostics'][0]['code'] == 'PARAMETER_BUDGET_EXCEEDED'
    for field in ('batch_size','activation_limit'):
        assert not ModelBuilder.preflight(builder.to_config(),**{field:0})['valid']


def test_g_old_conv_defaults_and_state_keys_are_preserved():
    legacy = dict(format='hyso.model_config',schema_version=1,kind='sequential_classifier',
                  input_shape=[4,5,7],num_classes=3,blocks=[dict(name='conv2d',params=dict(in_channels=4,out_channels=8))])
    model = ModelBuilder.from_config(legacy).build()
    assert list(model.state_dict()) == ['features.0.weight','features.0.bias','classifier.weight','classifier.bias']
    assert model.features[0].dilation == (1,1)
    assert model.features[0].kernel_size == (3,3) and model.features[0].padding == (1,1)
    assert ModelBuilder.preflight(legacy)['feature_shape'] == ['N',8,5,7]


def example_builder():
    return (ModelBuilder((3,64,64),5).add('conv2d',in_channels=3,out_channels=32)
            .add('batch_norm',num_features=32).add('gelu')
            .add('depthwise_conv2d',in_channels=32)
            .add('pointwise_conv2d',in_channels=32,out_channels=64)
            .add('group_norm',num_groups=8,num_channels=64).add('max_pool2d',kernel_size=2)
            .add('se',channels=64).add('adaptive_avg_pool2d',output_size=1))


def test_j_training_session_requested_chain_and_resume_boundary(tmp_path):
    from hyso import TrainingConfigError
    builder = example_builder()
    report = ModelBuilder.preflight(builder.to_config())
    assert report['feature_shape'] == ['N',64,1,1]
    data = TensorDataset(torch.randn(2,3,64,64),torch.tensor([0,1]))
    config = TrainingConfig(epochs=1,batch_size=2,optimizer='sgd')
    loader,_ = config.make_loaders(data)
    model = builder.build()
    result = config.prepare(model,loader).fit().to_dict()
    assert result['applied_updates'] == 1 and result['completed_history_epochs'] == 1
    for field in ('save_last_to','resume_from'):
        with pytest.raises(TrainingConfigError):
            TrainingConfig(epochs=1,batch_size=2,**{field:str(tmp_path/'state.pth')}).prepare(model,loader)


def test_k_cli_metadata_validation_summary_and_json_rejection(tmp_path,capsys):
    path = tmp_path/'architecture.json'
    example_builder().to_json(path)
    for args in (['show','conv_transpose2d'],['validate',str(path)],['summary',str(path)]):
        assert main(['models',*args,'--json']) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload['schema_version'] == 1 and payload['exit_code'] == 0
        if args[0] != 'show': assert payload['result']['validation']['layers'][0]['parameter_count'] > 0
    path.write_text('{"format":"hyso.model_config","format":"duplicate"}',encoding='utf-8')
    assert main(['models','validate',str(path),'--json']) == 2
    assert json.loads(capsys.readouterr().out)['diagnostics'][0]['code'] == 'INVALID_PARAMETER'
