"""Exercise the commercial wheel/sdist independently of checkout-only models."""

import zipfile
import tarfile

import pytest

from test_packaging import ROOT, MODULES, PROFILE, command
from test_packaging import distributions as distributions

pytestmark = pytest.mark.commercial_profile


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_commercial_archive_allowlist(distributions, kind):  # noqa: F811 - pytest fixture injection
    _, wheel, sdist = distributions
    if kind == "wheel":
        with zipfile.ZipFile(wheel) as archive:
            payload = {
                name: archive.read(name)
                for name in archive.namelist()
                if name.endswith(".py")
            }
        assert set(payload) == MODULES
    else:
        with tarfile.open(sdist) as archive:
            payload = {
                member.name.removeprefix("hyso-0.2.0/"): archive.extractfile(
                    member
                ).read()
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith(".py")
            }
        assert set(payload) == MODULES | {"setup.py"}
    assert not set(payload) & set(PROFILE["excluded_python_files"])
    for name in MODULES:
        assert payload[name] == (ROOT / name).read_bytes()


COMMERCIAL_SMOKE = r"""
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
    'pointwise_conv2d': dict(in_channels=4,out_channels=6),
    'depthwise_conv2d': dict(in_channels=4,depth_multiplier=2),
    'depthwise_separable_conv2d': dict(in_channels=4,out_channels=6,depth_multiplier=2),
    'spatial_separable_conv2d': dict(in_channels=4,out_channels=6,intermediate_channels=2),
    'factorized_conv2d': dict(in_channels=4,out_channels=6,intermediate_channels=2),
    'conv_transpose2d': dict(in_channels=4,out_channels=6,stride=2,output_padding=1),
    'channel_layer_norm': dict(channels=4),
    'instance_norm': dict(num_features=4,affine=True,track_running_stats=True),
    'leaky_relu': {}, 'prelu': dict(num_parameters=4), 'gelu': {}, 'mish': {}, 'hardsigmoid': {},
    'max_pool2d': dict(kernel_size=2), 'avg_pool2d': dict(kernel_size=3,padding=1),
    'adaptive_avg_pool2d': dict(output_size=[2,3]),
    'adaptive_max_pool2d': dict(output_size=[2,3]), 'global_avg_pool2d': {},
    'bottleneck_residual': dict(in_channels=4,out_channels=4,intermediate_channels=8),
    'preactivation_residual': dict(in_channels=4,out_channels=4,intermediate_channels=8),
    'wide_residual_block': dict(in_channels=4,out_channels=4,intermediate_channels=8),
    'grouped_bottleneck': dict(in_channels=4,out_channels=4,intermediate_channels=8,groups=2),
    'inverted_bottleneck': dict(in_channels=4,out_channels=4,expansion_ratio=2),
    'gated_residual': dict(in_channels=4,out_channels=4,intermediate_channels=8),
    'spatial_attention': dict(channels=4), 'channel_attention': dict(channels=4),
    'spatial_self_attention': dict(channels=4),
    'multihead_spatial_attention': dict(channels=4,num_heads=2),
    'dropout': dict(p=.2), 'dropout2d': dict(p=.2), 'drop_path': dict(p=.2), 'gaussian_noise': dict(std=.1),
}
from hyso.models.components import NEW_NAMES
from hyso.models.sampling import NEW_NAMES as SAMPLING_NAMES
assert set(samples) | {'modular_cnn'} | NEW_NAMES | SAMPLING_NAMES == {s.name for s in ModelRegistry().list()}
for dtype in (torch.float32, torch.float64):
    for name, options in samples.items():
        model = ModelRegistry().create(name, **options).to(dtype)
        inputs = torch.randn(2,4,8,8,dtype=dtype,requires_grad=True)
        output = model(inputs)
        report = ModelBuilder.preflight(ModelBuilder((4,8,8),3).add(name,**options).to_config())
        assert list(output.shape) == [2,*report['feature_shape'][1:]] and torch.isfinite(output).all()
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
                .add('depthwise_separable_conv2d',in_channels=4,out_channels=4)
                .add('channel_layer_norm',channels=4).add('gelu').add('avg_pool2d',kernel_size=3,padding=1,stride=1)
                .add('bottleneck_residual',in_channels=4,out_channels=4,intermediate_channels=8)
                .add('multihead_spatial_attention',channels=4,num_heads=2)
                .add('drop_path',p=.1)
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
new_layers = {
    'resize': dict(output_size=[6,10],mode='bilinear',align_corners=False),
    'pixel_shuffle': dict(upscale_factor=2), 'pixel_unshuffle': dict(downscale_factor=2),
    'flatten': {}, 'permute': {}, 'reshape': dict(shape=[2,8,16]),
}
assert set(new_layers) | {'classification_head','regression_head','embedding_head'} == SAMPLING_NAMES
for dtype in (torch.float32,torch.float64):
    for name,options in new_layers.items():
        builder = ModelBuilder.sequential((4,8,8),head='classification',num_classes=3,
                    pooling='none' if name=='flatten' else 'avg').add(name,**options)
        report = ModelBuilder.preflight(builder.to_config())
        model = builder.build().to(dtype)
        inputs = torch.randn(2,4,8,8,dtype=dtype,requires_grad=True)
        assert list(model.features(inputs).shape) == [2,*report['feature_shape'][1:]]
        output = model(inputs); output.square().sum().backward()
        assert torch.isfinite(inputs.grad).all()
        other = ModelBuilder.from_config(builder.to_config()).build().to(dtype)
        other.load_state_dict(model.state_dict(),strict=True)
    for head,params in [('classification',dict(num_classes=3)),('regression',dict(output_features=2)),
                        ('embedding',dict(embedding_dim=5,normalize=True))]:
        builder = ModelBuilder.sequential((4,8,8),head=head,**params)
        model = builder.build().to(dtype)
        inputs = torch.randn(2,4,8,8,dtype=dtype,requires_grad=True)
        optimizer = torch.optim.SGD(model.parameters(),lr=.01)
        output = model(inputs)
        torch.nn.functional.mse_loss(output,torch.randn_like(output)).backward()
        assert torch.isfinite(inputs.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        clone = ModelBuilder.from_config(builder.to_config()).build().to(dtype)
        clone.load_state_dict(model.state_dict(),strict=True)
        model.eval();clone.eval();assert torch.equal(model(inputs),clone(inputs))
from hyso import GraphModelBuilder
graph = (GraphModelBuilder({'image':[4,8,8]})
    .add('left','pointwise_conv2d',['image'],in_channels=4,out_channels=4)
    .add('right','conv2d',['image'],in_channels=4,out_channels=4)
    .add('concat','concat',['left','right'],axis=1)
    .add('project','pointwise_conv2d',['concat'],in_channels=8,out_channels=4)
    .add('sum','add',['project','image'])
    .add('weighted','weighted_sum',['sum','left'],weights=[.25,.75],learnable=True)
    .add('product','multiply',['weighted','right'])
    .add('classify','classification_head',['product'],in_features=4,num_classes=3)
    .add('regress','regression_head',['product'],in_features=4,output_features=2)
    .add('embed','embedding_head',['product'],in_features=4,embedding_dim=5,normalize=True)
    .outputs(classification='classify',regression='regress',embedding='embed',features='concat'))
report = graph.preview(batch_size=2)
assert report['valid'] and report['nodes']['concat']['output_shape'] == ['N',8,8,8]
model = graph.build()
optimizer = torch.optim.SGD(model.parameters(),lr=.01)
inputs = torch.randn(2,4,8,8,requires_grad=True)
outputs = model(inputs)
sum((value-torch.randn_like(value)).square().mean() for value in outputs.values()).backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
assert torch.isfinite(inputs.grad).all()
optimizer.step()
path = target.parent / ('graph-'+target.name+'.json')
graph.to_json(path)
clone = GraphModelBuilder.from_json(path).build()
clone.load_state_dict(model.state_dict(),strict=True)
model.eval();clone.eval()
for name,value in model(inputs).items(): assert torch.equal(value,clone(inputs)[name])
def no_allocate(*args,**kwargs): raise AssertionError('graph preview allocated weights')
original_module_init, original_empty = torch.nn.Module.__init__, torch.empty
torch.nn.Module.__init__ = no_allocate
torch.empty = no_allocate
assert GraphModelBuilder.preflight(graph.to_config())['valid']
stream = io.StringIO()
with redirect_stdout(stream): assert main(['models','preview',str(path),'--json']) == 0
assert json.loads(stream.getvalue())['result']['validation']['outputs']['embedding']['layout'] == 'NC'
torch.nn.Module.__init__, torch.empty = original_module_init, original_empty
from hyso.models.output import GraphTensorOutput
from hyso.model_checks import run_model_checks
from hyso.architecture_comparison import compare_architectures
from hyso.model_observations import diagnose_results
assert importlib.util.find_spec('hyso.studio') is None
selected = GraphTensorOutput(graph.build(), 'classification', task='multiclass').eval()
assert selected(torch.ones(2,4,8,8)).shape == (2,3)
checks = run_model_checks(graph.to_config(), tests=('structure','forward','backward','serialization'))
assert all(c['status']=='PASS' for c in checks['cases']), checks
assert len(compare_architectures([graph.to_config(),graph.to_config()])['columns']) == 2
assert diagnose_results([]) == []
print('commercial installed blocks/JSON/training/CLI/Python PASS')
"""


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_installed_commercial_core(distributions, kind):  # noqa: F811 - pytest fixture injection
    work, wheel, sdist = distributions
    target = work / ("commercial-" + kind)
    command(
        [
            "-m",
            "pip",
            "--isolated",
            "--no-cache-dir",
            "install",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--no-compile",
            "--target",
            str(target),
            str(wheel if kind == "wheel" else sdist),
        ],
        work,
    )
    assert "commercial installed blocks/JSON/training/CLI/Python PASS" in command(
        ["-c", COMMERCIAL_SMOKE, str(target)], work
    )
