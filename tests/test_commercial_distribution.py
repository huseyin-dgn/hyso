"""Exercise the commercial wheel/sdist independently of checkout-only models."""
import json
import zipfile
import tarfile

import pytest

pytestmark = pytest.mark.commercial_profile

from test_packaging import ROOT, MODULES, PROFILE, distributions, command


@pytest.mark.parametrize('kind', ['wheel', 'sdist'])
def test_commercial_archive_allowlist(distributions, kind):
    _, wheel, sdist = distributions
    if kind == 'wheel':
        with zipfile.ZipFile(wheel) as archive:
            payload = {name: archive.read(name) for name in archive.namelist()
                       if name.endswith('.py')}
        assert set(payload) == MODULES
    else:
        with tarfile.open(sdist) as archive:
            payload = {member.name.removeprefix('hyso-0.2.0/'): archive.extractfile(member).read()
                       for member in archive.getmembers()
                       if member.isfile() and member.name.endswith('.py')}
        assert set(payload) == MODULES | {'setup.py'}
    assert not set(payload) & set(PROFILE['excluded_python_files'])
    for name in MODULES:
        assert payload[name] == (ROOT / name).read_bytes()


COMMERCIAL_SMOKE = r'''
import ast, io, json, os, pathlib, sys, importlib.util
from contextlib import redirect_stdout
target = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
import torch
torch.set_num_threads(1)
def forbidden(*a, **kw):
    raise AssertionError('commercial CPU validation attempted CUDA work')
torch.cuda._lazy_init = forbidden
torch.cuda.is_available = lambda: False
import hyso
from hyso import ModelBuilder, ModelRegistry, TrainingConfig, ModularCNN
from hyso.models._profile import PROFILE
from hyso.cli import main
assert pathlib.Path(hyso.__file__).resolve().is_relative_to(target)
assert PROFILE == 'commercial'
os.environ['HYSO_DEVELOPMENT'] = '1'
assert PROFILE == 'commercial'
excluded = ('resnet','preact_resnet','wide_resnet','resnext','densenet','mobilenet_v2',
            'mobilenet_v3','efficientnet_v2','ghostnet','shufflenet_v2','repvgg','mobileone','fasternet')
for name in excluded:
    config = dict(format='hyso.model_config', schema_version=1,
                  kind='registered_model', name=name, params={})
    for operation in (lambda: ModelRegistry().get(name), lambda: ModelBuilder.from_config(config)):
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError('excluded model admitted: '+name)
for name in ('architectures','m3_residual','m3_attention','m4_blocks','m4_config','m4_registry',
             'm4_reparam','mobile_architectures','reparam_architectures','_development'):
    assert importlib.util.find_spec('hyso.models.'+name) is None
    try:
        __import__('hyso.models.'+name)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError('hidden development module imported')
for name in ('ResNet','DenseNet','MobileNetV2','GhostNet','RepVGG','MobileOne','FasterNet',
             'PartialConv','SimAM','TripletAttention','SelectiveKernelAttention','reparameterize_model'):
    assert not hasattr(hyso.models, name), name
samples = {
    'conv_norm_act': dict(in_channels=4,out_channels=4),
    'conv2d': dict(in_channels=4,out_channels=4),
    'residual': dict(in_channels=4,out_channels=4),
    'se': dict(channels=4), 'cbam': dict(channels=4), 'eca': dict(channels=4),
    'coordinate': dict(channels=4), 'batch_norm': dict(num_features=4),
    'group_norm': dict(num_groups=2,num_channels=4), 'relu': {}, 'silu': {},
    'hardswish': {}, 'identity': {}, 'channel_shuffle': dict(channels=4),
    'dropblock2d': dict(p=.2,block_size=3),
}
assert set(samples) | {'modular_cnn'} == {s.name for s in ModelRegistry().list()}
for dtype in (torch.float32, torch.float64):
    for name, options in samples.items():
        model = ModelRegistry().create(name, **options).to(dtype)
        inputs = torch.randn(2,4,8,8,dtype=dtype,requires_grad=True)
        output = model(inputs)
        assert output.shape == inputs.shape and torch.isfinite(output).all()
        output.square().mean().backward()
        assert torch.isfinite(inputs.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        stream = io.BytesIO()
        torch.save(model.state_dict(),stream)
        stream.seek(0)
        clone = ModelRegistry().create(name, **options).to(dtype)
        clone.load_state_dict(torch.load(stream,weights_only=True),strict=True)
        model.eval(); clone.eval()
        assert torch.equal(model(inputs),clone(inputs))
architecture = (ModelBuilder((4,8,8),3).add('conv_norm_act',in_channels=4,out_channels=4)
                .add('residual',in_channels=4,out_channels=4)
                .add('channel_shuffle',channels=4).add('dropblock2d',p=.1))
model = architecture.build()
from torch.utils.data import TensorDataset, DataLoader
loader = DataLoader(TensorDataset(torch.randn(2,4,8,8),torch.tensor([0,1])),batch_size=2)
result = TrainingConfig(epochs=1,batch_size=2,optimizer='sgd').prepare(model,loader).fit()
assert result.to_dict()['applied_updates'] == 1
clone = ModelBuilder.from_config(json.loads(json.dumps(architecture.to_config()))).build()
clone.load_state_dict(model.state_dict(),strict=True)
model.eval(); clone.eval()
inputs = torch.randn(2,4,8,8)
assert torch.equal(model(inputs),clone(inputs))
# The existing native Trainer imports EMA and loader-plan helpers at their original paths.
from hyso.models.WithSE import EMA
from hyso.models.bcam import _resume_loader_plan
assert _resume_loader_plan(loader)['batches'] == 1
assert EMA(ModularCNN(input_channels=4,num_classes=3,widths=(4,),repeats=(1,),strides=(1,))).ema_model
path = target.parent / ('commercial-'+target.name+'.json')
architecture.to_json(path)
for args in (['models','list'], ['models','show','residual'],
             ['models','validate',str(path)], ['models','summary',str(path)],
             ['models','inspect',str(path),'--execute']):
    stream = io.StringIO()
    with redirect_stdout(stream):
        assert main([*args,'--json']) == 0
    payload = json.loads(stream.getvalue())
    assert payload['exit_code'] == 0
assert not torch.cuda.is_initialized()
print('commercial installed blocks/JSON/training/CLI PASS')
'''


@pytest.mark.parametrize('kind', ['wheel', 'sdist'])
def test_installed_commercial_core(distributions, kind):
    work, wheel, sdist = distributions
    target = work / ('commercial-' + kind)
    command(['-m', 'pip', '--isolated', '--no-cache-dir', 'install', '--no-index',
             '--no-deps', '--no-build-isolation', '--no-compile', '--target', str(target),
             str(wheel if kind == 'wheel' else sdist)], work)
    assert 'commercial installed blocks/JSON/training/CLI PASS' in command(
        ['-c', COMMERCIAL_SMOKE, str(target)], work)
