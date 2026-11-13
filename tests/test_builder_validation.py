"""Allocation-free I/O diagnostics plus bounded CPU shape parity checks."""
import json

import pytest
import torch

from hyso import ModelBuilder, ModelRegistry, ModelConfigError
from hyso.cli import main
from test_packaging import distributions, command


def config():
    return dict(format='hyso.model_config', schema_version=1, kind='sequential_classifier',
                input_shape=[4,9,11], num_classes=3, blocks=[])


@pytest.fixture
def metadata_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('metadata validation allocated a module or queried CUDA')
    monkeypatch.setattr(torch.nn.Module, '__init__', forbidden)
    for name in ('_lazy_init', 'is_available', 'device_count'):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.mark.parametrize('name,params,output', [
    ('conv2d', dict(in_channels=4,out_channels=8,kernel_size=[3,5],stride=[2,3],padding=[1,2]), [8,5,4]),
    ('conv2d', dict(in_channels=4,out_channels=8,groups=2), [8,9,11]),
    ('conv2d', dict(in_channels=4,out_channels=8,groups=4), [8,9,11]),
    ('conv_norm_act', dict(in_channels=4,out_channels=8,stride=2), [8,5,6]),
    ('residual', dict(in_channels=4,out_channels=8,stride=[2,3]), [8,5,4]),
    ('se', dict(channels=4), [4,9,11]), ('cbam', dict(channels=4), [4,9,11]),
    ('eca', dict(channels=4), [4,9,11]), ('coordinate', dict(channels=4), [4,9,11]),
    ('batch_norm', dict(num_features=4), [4,9,11]),
    ('group_norm', dict(num_groups=2,num_channels=4), [4,9,11]),
    ('relu', {}, [4,9,11]), ('silu', {}, [4,9,11]), ('hardswish', {}, [4,9,11]),
    ('identity', {}, [4,9,11]), ('channel_shuffle', dict(channels=4), [4,9,11]),
    ('dropblock2d', dict(p=.2), [4,9,11]),
])
def test_symbolic_shape_without_weights(name,params,output,metadata_only):
    c = config()
    c['blocks'] = [dict(name=name,params=params)]
    result = ModelBuilder.preflight(c)
    assert result['valid'] and result['diagnostics'] == []
    assert result['input_shape'] == ['N',4,9,11]
    assert result['layers'][0]['output_shape'] == ['N',*output]
    assert result['feature_shape'] == ['N',*output]
    assert result['output_shape'] == ['N',3] and result['parameter_estimate'] > 0
    assert c['blocks'][0]['params'] == params


@pytest.mark.parametrize('shape',[None,[],[4,0,3],[4,-1,3],[0,3,3],[4,3,8193],
                                 [8193,3,3],[True,3,3],[4,2.5,3],[2,4,3,3]])
def test_invalid_input_shape(shape,metadata_only):
    c = config(); c['input_shape'] = shape
    report = ModelBuilder.preflight(c)
    assert not report['valid']
    assert report['diagnostics'][0]['code'] == 'INVALID_INPUT_SHAPE'
    assert report['diagnostics'][0]['actual'] == shape


@pytest.mark.parametrize('name,params,code',[
    ('conv2d',dict(in_channels=3,out_channels=4),'CHANNEL_MISMATCH'),
    ('conv2d',dict(in_channels=4,out_channels=4,kernel_size=0),'INVALID_PARAMETER'),
    ('conv2d',dict(in_channels=4,out_channels=4,stride=0),'INVALID_PARAMETER'),
    ('conv2d',dict(in_channels=4,out_channels=4,padding=-1),'INVALID_PARAMETER'),
    ('conv2d',dict(in_channels=4,out_channels=4,dilation=0),'INVALID_PARAMETER'),
    ('conv2d',dict(in_channels=4,out_channels=4,groups=3),'GROUP_DIVISIBILITY_ERROR'),
    ('group_norm',dict(num_channels=4,num_groups=3),'GROUP_DIVISIBILITY_ERROR'),
    ('channel_shuffle',dict(channels=4,groups=3),'GROUP_DIVISIBILITY_ERROR'),
    ('eca',dict(channels=3),'CHANNEL_MISMATCH'),
    ('residual',dict(in_channels=4,out_channels=8,attention=dict(name='se',params=dict(channels=4))),'CHANNEL_MISMATCH'),
    ('conv2d',dict(in_channels=4,out_channels=4,kernel_size=20,padding=0),'SPATIAL_DIMENSION_ERROR'),
    ('conv2d',dict(in_channels=4,out_channels=4,padding=8192),'SPATIAL_DIMENSION_ERROR'),
    ('modular_cnn',{},'INVALID_BLOCK_ORDER'),
    ('os.system',{},'UNSUPPORTED_COMPONENT'),
])
def test_structured_block_errors(name,params,code,metadata_only):
    c = config(); c['blocks'] = [dict(name=name,params=params)]
    report = ModelBuilder.preflight(c)
    assert not report['valid'] and len(report['diagnostics']) == 1
    error = report['diagnostics'][0]
    assert error['code'] == code and error['layer_index'] == 0
    assert error['message_tr'] and error['message_en'] and error['suggestion']
    assert error['expected'] is not None and error['actual'] is not None
    with pytest.raises(ModelConfigError) as raised:
        ModelBuilder.from_config(c)
    assert isinstance(raised.value, ValueError) and raised.value.to_dict() == error


def test_independent_errors_do_not_invent_downstream_shapes(metadata_only):
    c = config()
    c['blocks'] = [dict(name='conv2d',params=dict(in_channels=3,out_channels=4)),
                   dict(name='channel_shuffle',params=dict(channels=4,groups=3)),
                   dict(name='conv2d',params=dict(in_channels=99,out_channels=4))]
    result = ModelBuilder.preflight(c)
    assert [d['code'] for d in result['diagnostics']] == ['CHANNEL_MISMATCH','GROUP_DIVISIBILITY_ERROR']
    assert [d['layer_index'] for d in result['diagnostics']] == [0,1]
    assert result['layers'] == [] and result['output_shape'] is None


def test_legacy_exception_arguments(metadata_only):
    for args in ((),('legacy message',),('legacy message',42)):
        error = ModelConfigError(*args)
        assert error.args == args and str(error) == str(ValueError(*args))
        assert error.to_dict()['message_en'] == str(error)


@pytest.mark.parametrize('classes',[0,-1,True,8193])
def test_invalid_output_head(classes,metadata_only):
    c = config(); c['num_classes'] = classes
    assert ModelBuilder.preflight(c)['diagnostics'][0]['code'] == 'INVALID_OUTPUT_HEAD'


def test_budget_before_allocation_and_legacy_modular(metadata_only):
    c = config(); c['input_shape'][0] = 8192
    c['blocks'] = [dict(name='conv2d',params=dict(in_channels=8192,out_channels=8192))]
    assert ModelBuilder.preflight(c)['diagnostics'][0]['code'] == 'PARAMETER_BUDGET_EXCEEDED'
    legacy = ModelBuilder.modular(input_channels=1,num_classes=2,widths=(4,),repeats=(1,),strides=(1,))
    report = legacy.preflight(legacy.to_config())
    assert report['valid'] and report['shape_scope'] == 'configuration_only'
    assert report['input_shape'] == ['N',1,None,None]
    assert report['output_shape'] == ['N',2]


@pytest.mark.parametrize('raw',['[]','{"schema_version":true}',
                              '{"format":"a","format":"b"}','{"input_shape":NaN}',
                              '{"module":"os","callable":"system"}'])
def test_invalid_json_cli_preserves_exit_and_developer_detail(raw,tmp_path,capsys,metadata_only):
    path = tmp_path/'bad.json'; path.write_text(raw,encoding='utf-8')
    assert main(['models','validate',str(path),'--json']) == 2
    captured = capsys.readouterr(); payload = json.loads(captured.out)
    assert payload['format'] == 'hyso.cli' and payload['schema_version'] == 1
    assert payload['error'] and payload['diagnostics'][0]['message_en']
    assert 'Traceback' not in captured.err


def test_cli_summary_and_channel_diagnostic(tmp_path,capsys,metadata_only):
    path = tmp_path/'model.json'; path.write_text(json.dumps(config()),encoding='utf-8')
    for action in ('validate','summary'):
        assert main(['models',action,str(path),'--json']) == 0
        assert json.loads(capsys.readouterr().out)['result']['validation']['output_shape'] == ['N',3]
    c = config(); c['blocks'] = [dict(name='eca',params=dict(channels=3))]
    path.write_text(json.dumps(c),encoding='utf-8')
    assert main(['models','validate',str(path),'--json']) == 2
    output = capsys.readouterr()
    error = json.loads(output.out)['diagnostics'][0]
    assert error['code'] == 'CHANNEL_MISMATCH' and error['expected'] == 3 and error['actual'] == 4
    assert 'kanal' in output.err and 'Traceback' not in output.err


@pytest.mark.parametrize('name,params',[
    ('conv2d',dict(in_channels=4,out_channels=8,groups=4,stride=2)),
    ('conv_norm_act',dict(in_channels=4,out_channels=8,kernel_size=[3,5],padding=[1,2],stride=[2,3])),
    ('residual',dict(in_channels=4,out_channels=8,stride=[2,3])),
])
def test_cpu_shape_matches_preflight_and_state_roundtrip(name,params):
    builder = ModelBuilder((4,9,11),3).add(name,**params)
    report = builder.preflight(builder.to_config())
    model = builder.build().eval()
    with torch.inference_mode():
        assert list(model.features(torch.zeros(2,4,9,11)).shape) == [2,*report['feature_shape'][1:]]
        assert model(torch.zeros(2,4,9,11)).shape == (2,3)
    clone = ModelBuilder.from_config(builder.to_config()).build().eval()
    clone.load_state_dict(model.state_dict(),strict=True)
    assert torch.equal(model(torch.ones(2,4,9,11)),clone(torch.ones(2,4,9,11)))


def test_runtime_nchw_and_dtype_diagnostics():
    model = ModelBuilder((4,2,2),3).build()
    for shape in ((4,2,2),(0,4,2,2),(1,3,2,2),(1,4,1,2)):
        with pytest.raises(ModelConfigError) as raised:
            model(torch.zeros(shape))
        assert raised.value.code == 'INVALID_INPUT_SHAPE'
        assert raised.value.expected == ['N',4,2,2]
    with pytest.raises(ModelConfigError) as raised:
        model(torch.ones(1,4,2,2,dtype=torch.float64))
    assert raised.value.actual['dtype'] == 'torch.float64'


def test_symbolic_batch_keeps_native_statistics_error_context():
    builder = ModelBuilder((2,1,1),3).add('batch_norm',num_features=2)
    assert ModelBuilder.preflight(builder.to_config())['valid']
    with pytest.raises(ModelConfigError) as raised:
        builder.build()(torch.ones(1,2,1,1))
    assert raised.value.layer_index == 0 and raised.value.__cause__ is not None


@pytest.mark.parametrize('kind',['wheel','sdist'])
def test_installed_commercial_preflight(distributions,kind):
    work, wheel, sdist = distributions
    target = work / ('validation-'+kind)
    command(['-m','pip','install','--no-index','--no-deps','--no-build-isolation',
             '--no-compile','--target',str(target),str(wheel if kind=='wheel' else sdist)],work)
    code = r'''
import sys, pathlib, json, io
from contextlib import redirect_stdout
target = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0,str(target))
from hyso import ModelBuilder, ModelRegistry
from hyso.models._profile import PROFILE
from hyso.cli import main
import hyso, torch
assert pathlib.Path(hyso.__file__).resolve().is_relative_to(target)
assert PROFILE == 'commercial' and len(ModelRegistry().list()) == 34
def forbidden(*a,**kw):
    raise AssertionError('validation attempted allocation or CUDA')
torch.nn.Module.__init__ = forbidden
torch.cuda._lazy_init = forbidden
torch.cuda.is_available = forbidden
builder = ModelBuilder((4,9,11),3).add('conv2d',in_channels=4,out_channels=8,groups=4,stride=2)
assert ModelBuilder.preflight(builder.to_config())['feature_shape'] == ['N',8,5,6]
config = builder.to_config()
config['blocks'][0]['params']['in_channels'] = 3
error = ModelBuilder.preflight(config)['diagnostics'][0]
assert error['code'] == 'GROUP_DIVISIBILITY_ERROR' and error['layer_index'] == 0
assert error['message_tr'] and error['message_en'] and error['suggestion']
path = target.parent / ('preflight-'+target.name+'.json')
builder.to_json(path)
stream = io.StringIO()
with redirect_stdout(stream):
    assert main(['models','summary',str(path),'--json']) == 0
assert json.loads(stream.getvalue())['result']['validation']['output_shape'] == ['N',3]
assert not torch.cuda.is_initialized()
print('commercial preflight PASS')
'''
    assert 'commercial preflight PASS' in command(['-c',code,str(target)],work)
