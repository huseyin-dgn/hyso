"""CPU/metadata contracts for the thin Pythonic builder convenience methods."""
import copy
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import typing
import venv

import pytest
import torch

from hyso import GraphModelBuilder, ModelBuilder, ModelConfigError, ModelRegistry
from test_packaging import distributions

ROOT=Path(__file__).resolve().parents[1]
NAMES=('conv2d','conv_norm_act','relu','silu','gelu','batch_norm','group_norm',
       'max_pool2d','avg_pool2d','adaptive_avg_pool2d','dropout')
REQUIRED={
    'conv2d':dict(in_channels=4,out_channels=4),
    'conv_norm_act':dict(in_channels=4,out_channels=4),
    'batch_norm':dict(num_features=4), 'group_norm':dict(num_groups=2,num_channels=4),
    'max_pool2d':dict(kernel_size=2), 'avg_pool2d':dict(kernel_size=2),
    'adaptive_avg_pool2d':dict(output_size=2),
}


@pytest.fixture(autouse=True)
def cpu_only_no_training(monkeypatch):
    from hyso import Trainer,TrainingSession
    torch.set_num_threads(1)
    def fence(label):
        def forbidden(*a,**kw):pytest.fail('Pythonic API test started '+label)
        return forbidden
    for name in ('init','_lazy_init','is_available','device_count','synchronize'):
        monkeypatch.setattr(torch.cuda,name,fence(name))
    monkeypatch.setattr(Trainer,'fit',fence('Trainer.fit'))
    monkeypatch.setattr(TrainingSession,'fit',fence('TrainingSession.fit'))


def builders(cls):
    return cls({'image':[4,8,8]}) if cls is GraphModelBuilder else cls((4,8,8),2)


def append(builder,name,params,*,wrapper):
    if isinstance(builder,GraphModelBuilder):
        result=(getattr(builder,name)('node',['image'],**params) if wrapper else
                builder.add('node',name,['image'],**params))
        result.outputs(features='node')
    else:result=getattr(builder,name)(**params) if wrapper else builder.add(name,**params)
    assert result is builder
    return result


@pytest.mark.parametrize('cls',[ModelBuilder,GraphModelBuilder])
@pytest.mark.parametrize('name',NAMES)
def test_explicit_signature_defaults_and_config_equivalence(cls,name,monkeypatch):
    def forbidden(*a,**kw):pytest.fail('config-only wrapper allocated a module')
    monkeypatch.setattr(torch.nn.Module,'__init__',forbidden)
    method=getattr(cls,name);sig=inspect.signature(method)
    expected={p.name:p for p in ModelRegistry().get(name).parameters}
    component_params={n:p for n,p in sig.parameters.items() if n not in ('self','node_id','inputs')}
    assert set(component_params)==set(expected)
    for n,p in component_params.items():
        assert p.kind is p.KEYWORD_ONLY
        inferred = n == {'conv2d':'in_channels', 'batch_norm':'num_features'}.get(name)
        assert p.default==(None if inferred else inspect.Parameter.empty if expected[n].required else expected[n].default)
    assert all(p.kind not in (p.VAR_KEYWORD,p.VAR_POSITIONAL) for p in sig.parameters.values())
    hints=typing.get_type_hints(method)
    assert hints['return'] is cls
    assert set(hints)==set(sig.parameters)-{'self'}|{'return'}
    params=REQUIRED.get(name,{})
    new=append(builders(cls),name,params,wrapper=True)
    old=append(builders(cls),name,params,wrapper=False)
    assert new.to_config()==old.to_config()
    assert type(new).from_config(new.to_config()).to_config()==old.to_config()


@pytest.mark.parametrize('cls',[ModelBuilder,GraphModelBuilder])
@pytest.mark.parametrize('name,params',[
    ('conv2d',dict(in_channels=4,out_channels=4,kernel_size=(1,3),stride=[1,2],padding=(0,1),groups=2,bias=False,dilation=[1,1])),
    ('conv_norm_act',dict(in_channels=4,out_channels=4,kernel_size=[1,3],stride=(1,1),padding=[0,1],groups=2,norm=None,activation=None)),
    ('batch_norm',dict(num_features=4,eps=1e-4,momentum=None,affine=False,track_running_stats=False)),
    ('group_norm',dict(num_groups=2,num_channels=4,eps=1e-4,affine=False)),
    ('max_pool2d',dict(kernel_size=(2,3),stride=[1,2],padding=0,dilation=(1,1),ceil_mode=True)),
    ('avg_pool2d',dict(kernel_size=[2,3],stride=(1,2),padding=0,ceil_mode=True,count_include_pad=False)),
    ('adaptive_avg_pool2d',dict(output_size=(2,3))),('dropout',dict(p=1.0)),('gelu',dict(approximate='tanh')),
])
def test_every_option_is_forwarded_without_reinterpretation(cls,name,params):
    assert append(builders(cls),name,params,wrapper=True).to_config()==append(builders(cls),name,params,wrapper=False).to_config()


@pytest.mark.parametrize('cls',[ModelBuilder,GraphModelBuilder])
@pytest.mark.parametrize('name,params',[
    ('conv2d',dict(in_channels=True,out_channels=4)),
    ('conv_norm_act',dict(in_channels=4,out_channels=4,norm='layer')),
    ('gelu',dict(approximate='fast')),('batch_norm',dict(num_features=4,momentum=2.0)),
    ('max_pool2d',dict(kernel_size=0)),('dropout',dict(p=-0.1)),
])
def test_invalid_values_use_the_existing_validator_and_keep_state(cls,name,params):
    new,old=builders(cls),builders(cls);before=new.to_config()
    with pytest.raises(ModelConfigError) as actual:append(new,name,params,wrapper=True)
    with pytest.raises(ModelConfigError) as expected:append(old,name,params,wrapper=False)
    assert str(actual.value)==str(expected.value)
    assert new.to_config()==old.to_config()==before


@pytest.mark.parametrize('cls',[ModelBuilder,GraphModelBuilder])
def test_unknown_keywords_missing_arguments_and_existing_fixed_profile_guard(cls):
    b=builders(cls);prefix=('node',['image']) if cls is GraphModelBuilder else ()
    for name,kw in [('relu',dict(inplace=True)),('max_pool2d',dict(kernel_size=2,return_indices=True)),
                    ('avg_pool2d',dict(kernel_size=2,divisor_override=2)),('conv2d',{})]:
        before=b.to_config()
        with pytest.raises(TypeError):getattr(b,name)(*prefix,**kw)
        assert b.to_config()==before
    fixed=ModelBuilder.modular(input_channels=4,num_classes=2,widths=(4,),repeats=(1,),strides=(1,))
    with pytest.raises(ModelConfigError):fixed.relu()


def test_literal_options_exactly_match_registry_choices_and_runtime_values():
    registry=ModelRegistry()
    for cls in (ModelBuilder,GraphModelBuilder):
        for name,field in [('conv_norm_act','norm'),('conv_norm_act','activation'),('gelu','approximate')]:
            hint=typing.get_type_hints(getattr(cls,name))[field]
            allowed=next(p.choices for p in registry.get(name).parameters if p.name==field)
            assert typing.get_origin(hint) is typing.Literal and set(typing.get_args(hint))==set(allowed)
            for choice in allowed:
                params=dict(REQUIRED.get(name,{}),**{field:choice})
                assert append(builders(cls),name,params,wrapper=True).to_config()==append(builders(cls),name,params,wrapper=False).to_config()


def test_graph_names_connections_outputs_and_legacy_fusions_stay_explicit():
    g=GraphModelBuilder({'image':[4,8,8]}).conv2d('left',['image'],in_channels=4,out_channels=4)
    g.relu('right',['left']).add('merge','add',['left','right']).outputs(features='merge')
    assert g.preview()['valid']
    before=g.to_config()
    with pytest.raises(ModelConfigError):g.relu('left',['image'])
    assert g.to_config()==before
    bad=GraphModelBuilder({'image':[4,8,8]}).relu('node',['missing']).outputs(features='node')
    assert not bad.preview()['valid']  # No invented connection or automatic topology repair.


def test_explicit_schema2_heads_still_use_the_existing_add_contract():
    new=ModelBuilder.sequential((4,8,8),num_classes=2)
    old=ModelBuilder.sequential((4,8,8),num_classes=2)
    new.conv2d(in_channels=4,out_channels=4).relu().adaptive_avg_pool2d(output_size=1)
    old.add('conv2d',in_channels=4,out_channels=4).add('relu').add('adaptive_avg_pool2d',output_size=1)
    assert new.to_config()['schema_version']==2
    assert new.to_config()==old.to_config()


@pytest.mark.parametrize('graph',[False,True])
def test_cpu_model_and_state_dict_parity_and_json_roundtrip(graph,tmp_path):
    cls=GraphModelBuilder if graph else ModelBuilder
    new=append(builders(cls),'conv2d',REQUIRED['conv2d'],wrapper=True)
    old=append(builders(cls),'conv2d',REQUIRED['conv2d'],wrapper=False)
    path=tmp_path/'model.json';new.to_json(path)
    assert cls.from_json(path).to_config()==old.to_config()
    a,b=new.build().eval(),old.build().eval()
    assert list(a.state_dict())==list(b.state_dict())
    b.load_state_dict(copy.deepcopy(a.state_dict()),strict=True)
    x=torch.ones(2,4,8,8,device='cpu')
    with torch.inference_mode():
        ay,by=(a({'image':x})['features'],b({'image':x})['features']) if graph else (a(x),b(x))
    assert torch.equal(ay,by) and all(p.device.type=='cpu' for p in a.parameters())


def test_three_examples_only_build_and_prepare(monkeypatch):
    def forbidden(*a,**kw):pytest.fail('example ran a model while only building/preparing')
    monkeypatch.setattr(torch.nn.Module,'_call_impl',forbidden)
    example=runpy.run_path(str(ROOT/'examples/models/pythonic_builders.py'))
    cnn=example['simple_cnn']();graph=example['graph_model']();session=example['prepare_training']()
    assert cnn.input_shape==(1,8,8)
    assert graph.to_config()['outputs']=={'logits':'head'}
    assert session.settings()['state']=='prepared'
    assert len(session.train_loader)==2 and session.val_loader is None


def test_new_methods_have_real_guide_topics_and_unchanged_four_language_registry():
    from hyso_guide import topic_for_api,guide_uri
    topics=json.loads((ROOT/'hyso/data/user-guide-topics.json').read_text(encoding='utf-8'))
    content=json.loads((ROOT/'docs/user-guide/content.json').read_text(encoding='utf-8'))
    assert len(topics['topics'])==len(content['topics']) and topics['languages']==['tr','en','de','es']
    for cls in (ModelBuilder,GraphModelBuilder):
        for name in NAMES:
            symbol='hyso.'+cls.__name__+'.'+name;topic=topic_for_api(symbol)
            assert topic.startswith('components.') and 'HYSo Guide: '+topic in inspect.getdoc(getattr(cls,name))
            for lang in topics['languages']:assert guide_uri(topic,lang).endswith('?lang='+lang+'#'+topic)


def test_jedi_completes_methods_and_displays_real_named_parameters():
    jedi=pytest.importorskip('jedi')
    prefix='from hyso import ModelBuilder, GraphModelBuilder\nb = ModelBuilder((4,8,8),2)\ng = GraphModelBuilder({"image":[4,8,8]})\n'
    for variable,cls in [('b',ModelBuilder),('g',GraphModelBuilder)]:
        completions=jedi.Script(prefix+variable+'.',path=str(ROOT/'completion_probe.py')).complete()
        assert set(NAMES).issubset({c.name for c in completions})
        signatures=jedi.Script(prefix+variable+'.conv2d(',path=str(ROOT/'completion_probe.py')).get_signatures()
        assert signatures and {p.name for p in signatures[0].params}==set(inspect.signature(cls.conv2d).parameters)-{'self'}
        assert 'kernel_size' in signatures[0].to_string() and 'padding' in signatures[0].to_string()
        arguments="'stem', ['image'], " if variable=='g' else ''
        signatures=jedi.Script(prefix+variable+'.conv_norm_act('+arguments,path=str(ROOT/'completion_probe.py')).get_signatures()
        assert signatures and 'Literal' in signatures[0].to_string()


def test_clean_wheel_wrappers_examples_and_offline_help(distributions,tmp_path):
    work,wheel,_=distributions
    target=tmp_path/'installed';venv.EnvBuilder(with_pip=False,system_site_packages=True).create(target)
    exe=target/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    env=dict(os.environ,PIP_CONFIG_FILE=os.devnull,PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8')
    env.pop('PYTHONPATH',None)
    def run(argv):
        p=subprocess.run(argv,cwd=work,env=env,capture_output=True,text=True,encoding='utf-8',timeout=120)
        assert p.returncode==0,p.stdout+p.stderr
        return p.stdout
    run([sys.executable,'-m','pip','--python',str(exe),'install','--no-index','--no-deps',str(wheel)])
    code='''import pathlib,sys,typing,inspect
import torch
torch.set_num_threads(1)
def fence(label):
 def forbidden(*a,**k):raise AssertionError(label)
 return forbidden
for name in ("init","_lazy_init","is_available","device_count","synchronize"):
 setattr(torch.cuda,name,fence(name))
import hyso
from hyso import ModelBuilder,GraphModelBuilder,Trainer,TrainingSession
assert pathlib.Path(hyso.__file__).is_relative_to(pathlib.Path(sys.prefix))
assert (pathlib.Path(hyso.__file__).parent/"py.typed").is_file()
Trainer.fit=fence("fit")
TrainingSession.fit=fence("session.fit")
names=WRAPPERS
for cls in (ModelBuilder,GraphModelBuilder):
 for name in names:
  assert callable(getattr(cls,name))
  assert typing.get_type_hints(getattr(cls,name))["return"] is cls
assert inspect.signature(ModelBuilder.conv2d).parameters["padding"].default==1
from hyso_guide import topic_for_api,show_topic
assert topic_for_api("hyso.ModelBuilder.conv2d")=="components.convolution"
assert show_topic("models.graph-builder","de")["lang"]=="de"
ns={"__name__":"installed_examples"}
exec(compile(EXAMPLE,"pythonic_builders.py","exec"),ns)
torch.nn.Module._call_impl=fence("unexpected forward")
assert ns["simple_cnn"]() is not None
assert ns["graph_model"]() is not None
assert ns["prepare_training"]().settings()["state"]=="prepared"
from hyso.cli import main
assert main(["docs","show","components.convolution","--lang","tr","--json"])==0
'''
    code='WRAPPERS='+repr(NAMES)+'\nEXAMPLE='+repr((ROOT/'examples/models/pythonic_builders.py').read_text(encoding='utf-8'))+'\n'+code
    assert 'components.convolution' in run([str(exe),'-I','-B','-c',code])
