"""Complete Registry signatures/configs and profile-aware installed completion, CPU only."""
import ast
import collections
import copy
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import tempfile
import typing
import zipfile

import pytest
import torch
from hyso import ModelBuilder,GraphModelBuilder,ModelRegistry,ModelConfigError
from test_packaging import distributions

ROOT=Path(__file__).resolve().parents[1]
INDEX=json.loads((ROOT/'hyso/data/api-topics.json').read_text(encoding='utf-8'))
INVENTORY=INDEX['registry_completion']['components']
FUSIONS=INDEX['registry_completion']['fusions']
SURFACES=[(cls,name) for cls,key in ((ModelBuilder,'sequential'),(GraphModelBuilder,'graph'))
          for name,row in INVENTORY.items() if row[key]]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    torch.set_num_threads(1)
    def fence(name):
        def forbidden(*args,**kwargs):pytest.fail('Registry completion started '+name)
        return forbidden
    for name in ('init','_lazy_init','is_available','device_count','synchronize'):
        monkeypatch.setattr(torch.cuda,name,fence(name))
    from hyso import Trainer,TrainingSession,BenchmarkRunner
    for cls,name in ((Trainer,'fit'),(TrainingSession,'fit'),(BenchmarkRunner,'run')):
        monkeypatch.setattr(cls,name,fence(name))


def arguments(name):
    spec=ModelRegistry().get(name)
    values=dict(in_channels=16,out_channels=16,channels=16,num_features=16,num_channels=16,
                num_groups=4,intermediate_channels=16,in_features=16,num_classes=3,
                embedding_dim=4,output_features=3,hidden_channels=32,kernel_size=2,
                output_size=(4,4),upscale_factor=2,downscale_factor=2,shape=(16,8,8))
    args={p.name:values[p.name] for p in spec.parameters if p.required}
    if name=='resize':args['output_size']=(4,4)
    if name=='resnext_bottleneck':args['out_channels']=256
    return args


def configurations(cls,name,args):
    row=INVENTORY[name]
    if row['category']=='model':
        new=getattr(cls,name)(**args)
        old=cls.modular(**args) if name=='modular_cnn' else cls.named(name,**args)
    elif cls is GraphModelBuilder:
        new=cls({'image':[16,8,8]});old=cls({'image':[16,8,8]})
        assert getattr(new,name)('node',['image'],**args) is new
        old.add('node',name,['image'],**args)
        new.outputs(features='node');old.outputs(features='node')
    else:
        def fresh():
            return cls.sequential((16,8,8),num_classes=3,pooling='none') if name=='flatten' else cls((16,8,8),3)
        new,old=fresh(),fresh();assert getattr(new,name)(**args) is new;old.add(name,**args)
    return new,old


@pytest.mark.parametrize('cls,name',SURFACES,ids=lambda x:x.__name__ if isinstance(x,type) else x)
def test_every_applicable_signature_and_real_config_equivalence(cls,name,monkeypatch):
    def forbidden(*a,**kw):pytest.fail('config wrapper allocated a torch module')
    monkeypatch.setattr(torch.nn.Module,'__init__',forbidden)
    spec=ModelRegistry().get(name);method=getattr(cls,name);signature=inspect.signature(method)
    actual={n:p for n,p in signature.parameters.items() if n not in ('self','node_id','inputs')}
    assert set(actual)=={p.name for p in spec.parameters}
    for param in spec.parameters:
        assert actual[param.name].kind is inspect.Parameter.KEYWORD_ONLY
        inferred = param.name == {'conv2d':'in_channels', 'batch_norm':'num_features',
                                 **({'classification_head':'in_features'} if cls is GraphModelBuilder else {})}.get(name)
        assert actual[param.name].default==(None if inferred else inspect.Parameter.empty if param.required else param.default)
    assert not any(p.kind in (p.VAR_KEYWORD,p.VAR_POSITIONAL) for p in signature.parameters.values())
    hints=typing.get_type_hints(method);assert hints['return'] is cls
    assert set(hints)==set(signature.parameters)-{'self'}|{'return'}
    for param in spec.parameters:
        if param.kind=='choice':assert set(typing.get_args(hints[param.name]))==set(param.choices)
        elif param.kind in ('positive_int','odd_int'):
            inferred = param.name == {'conv2d':'in_channels', 'batch_norm':'num_features',
                                     **({'classification_head':'in_features'} if cls is GraphModelBuilder else {})}.get(name)
            assert hints[param.name] == (int | None if inferred else int)
    new,old=configurations(cls,name,arguments(name));assert new.to_config()==old.to_config()
    assert cls.from_config(new.to_config()).to_config()==old.to_config()


@pytest.mark.parametrize('cls,name',SURFACES,ids=lambda x:x.__name__ if isinstance(x,type) else x)
def test_every_wrapper_rejects_bad_input(cls,name):
    params=ModelRegistry().get(name).parameters
    if not params:
        b=cls({'image':[16,8,8]}) if cls is GraphModelBuilder else cls((16,8,8),3)
        with pytest.raises(TypeError):getattr(b,name)(*(['node',['image']] if cls is GraphModelBuilder else []),unsupported_option=True)
        return
    args=arguments(name);args[params[0].name]='invalid-value'
    with pytest.raises(ModelConfigError):configurations(cls,name,args)


def test_inventory_counts_suitability_profiles_and_docs():
    registry=ModelRegistry();assert set(INVENTORY)=={s.name for s in registry.list()}
    assert len(INVENTORY)==85
    assert sum('commercial' in r['profiles'] for r in INVENTORY.values())==57
    assert sum(bool(r['sequential']) for r in INVENTORY.values())==82
    assert sum(bool(r['graph']) for r in INVENTORY.values())==72
    assert sum(bool(r['sequential']) and 'commercial' in r['profiles'] for r in INVENTORY.values())==54
    assert sum(bool(r['graph']) and 'commercial' in r['profiles'] for r in INVENTORY.values())==56
    from hyso_guide import topic_for_api
    for name,row in INVENTORY.items():
        assert row['sequential'] or row['graph']
        expected=[dict(name=p.name,kind=p.kind,required=p.required,default=p.default,choices=list(p.choices)) for p in registry.get(name).parameters]
        assert row['parameters']==json.loads(json.dumps(expected))
        for cls,key in ((ModelBuilder,'sequential'),(GraphModelBuilder,'graph')):
            if row[key]:
                method=getattr(cls,name);topic=topic_for_api('hyso.'+cls.__name__+'.'+name)
                assert topic==row['topic'] and 'HYSo Guide: '+topic in inspect.getdoc(method)
            else:assert not hasattr(cls,name)
    assert INVENTORY['flatten']['sequential']['restriction']=='terminal schema2 feature block only'
    assert set(FUSIONS)=={'add','concat','multiply','weighted_sum'}


@pytest.mark.parametrize('component',FUSIONS)
def test_fusions_preserve_add_topology_validation_and_cpu_state(component):
    def fresh():
        return (GraphModelBuilder({'image':[4,3,3]})
                .add('left','identity',['image']).add('right','identity',['image']))
    args=dict(weights=(0.25,0.75),learnable=True) if component=='weighted_sum' else dict(axis=1) if component=='concat' else {}
    new,old=fresh(),fresh();assert getattr(new,FUSIONS[component])('merge',['left','right'],**args) is new
    old.add('merge',component,['left','right'],**args)
    new.outputs(features='merge');old.outputs(features='merge');assert new.to_config()==old.to_config()
    a,b=new.build().eval(),old.build().eval();b.load_state_dict(copy.deepcopy(a.state_dict()),strict=True)
    x={'image':torch.ones(1,4,3,3,device='cpu')}
    with torch.inference_mode():assert torch.equal(a(x)['features'],b(x)['features'])
    broken=fresh();getattr(broken,FUSIONS[component])('merge',['left'],**args);broken.outputs(features='merge')
    assert not broken.preview()['valid']


def setup_helpers():
    tree=ast.parse((ROOT/'setup.py').read_text(encoding='utf-8'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('commercial_payload','write_commercial')]
    namespace=dict(json=json,Path=Path,tempfile=tempfile)
    exec(compile(ast.Module(body=functions,type_ignores=[]),'setup.py','exec'),namespace)
    return namespace


def test_packaging_transform_is_idempotent_and_never_edits_a_source_hardlink(tmp_path):
    helper=setup_helpers()
    for name in ('hyso/models/builder.py','hyso/models/graph.py','hyso/data/api-topics.json'):
        payload=(ROOT/name).read_bytes();projected=helper['commercial_payload'](name,payload)
        assert helper['commercial_payload'](name,projected)==projected
    source=tmp_path/'source.py';target=tmp_path/'package.py';payload=(ROOT/'hyso/models/builder.py').read_bytes();source.write_bytes(payload)
    os.link(source,target);helper['write_commercial'](target,'hyso/models/builder.py')
    assert source.read_bytes()==payload and target.read_bytes()!=payload
    assert not list(tmp_path.glob('package.py.*'))


def test_jedi_source_discovers_all_applicable_methods_and_fusions():
    jedi=pytest.importorskip('jedi')
    prefix='from hyso import ModelBuilder, GraphModelBuilder\nb=ModelBuilder((16,8,8),3)\ng=GraphModelBuilder({"image":[16,8,8]})\n'
    for variable,cls,key in (('b',ModelBuilder,'sequential'),('g',GraphModelBuilder,'graph')):
        expected={n for n,row in INVENTORY.items() if row[key]}
        if key=='graph':expected|=set(FUSIONS.values())
        names={c.name for c in jedi.Script(prefix+variable+'.',path=str(ROOT/'completion_probe.py')).complete()}
        assert expected.issubset(names),expected-names
        for name in expected:
            signatures=jedi.Script(prefix+variable+'.'+name+'(',path=str(ROOT/'completion_probe.py')).get_signatures()
            assert signatures,name
            assert {p.name for p in signatures[0].params}==set(inspect.signature(getattr(cls,name)).parameters)-{'self'},name


@pytest.mark.parametrize('name',['advanced_cnn','fused_graph'])
def test_advanced_examples_cpu_forward_without_training(name):
    functions=runpy.run_path(str(ROOT/'examples/models/pythonic_builders.py'))
    model=functions[name]().eval();x=torch.ones(2,16,8,8,device='cpu')
    with torch.inference_mode():output=model({'image':x})['logits'] if name=='fused_graph' else model(x)
    assert output.shape==(2,3) and torch.isfinite(output).all()
    assert all(p.device.type=='cpu' for p in model.parameters())


def test_commercial_wheel_and_sdist_runtime_static_completion_and_help(distributions,tmp_path):
    work,wheel,sdist=distributions
    # All unsupported definitions must be physically absent, not only hidden by a runtime check.
    helper=setup_helpers()['commercial_payload']
    for archive in (wheel,sdist):
        if archive.suffix=='.whl':
            with zipfile.ZipFile(archive) as z:payloads={n:z.read(n) for n in ('hyso/models/builder.py','hyso/models/graph.py','hyso/data/api-topics.json')}
        else:
            with tarfile.open(archive) as z:payloads={n:z.extractfile('hyso-0.2.0/'+n).read() for n in ('hyso/models/builder.py','hyso/models/graph.py','hyso/data/api-topics.json')}
        for n,payload in payloads.items():assert payload==helper(n,(ROOT/n).read_bytes())
        assert len(json.loads(payloads['hyso/data/api-topics.json'])['registry_completion']['components'])==57
    for kind,archive in (('wheel',wheel),('sdist',sdist)):
        target=tmp_path/kind
        p=subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--no-build-isolation','--no-compile','--target',str(target),str(archive)],cwd=work,capture_output=True,text=True,timeout=120)
        assert p.returncode==0,p.stdout+p.stderr
        code='''import pathlib,sys,typing,json,inspect
sys.path.insert(0,sys.argv[1])
import torch
def fence(label):
 def forbidden(*a,**k):raise AssertionError(label)
 return forbidden
for name in ("init","_lazy_init","is_available","device_count","synchronize"):setattr(torch.cuda,name,fence(name))
torch.nn.Module.__init__=fence("unexpected model construction")
from hyso import ModelBuilder,GraphModelBuilder,ModelRegistry
import hyso
assert pathlib.Path(hyso.__file__).is_relative_to(pathlib.Path(sys.argv[1]))
from hyso.models._profile import PROFILE
assert PROFILE=="commercial"
from hyso_guide import topic_for_api,show_topic
assert len(ModelRegistry().list())==57
assert {s.name for s in ModelRegistry().list()}=={n for n,r in INVENTORY.items() if "commercial" in r["profiles"]}
for cls,key in ((ModelBuilder,"sequential"),(GraphModelBuilder,"graph")):
 for name,row in INVENTORY.items():
  expected=bool(row[key]) and "commercial" in row["profiles"]
  assert hasattr(cls,name)==expected,(cls,name)
  if expected:
   assert typing.get_type_hints(getattr(cls,name))["return"] is cls
   assert topic_for_api("hyso."+cls.__name__+"."+name)==row["topic"]
for name in FUSIONS.values():assert hasattr(GraphModelBuilder,name)
try:topic_for_api("hyso.ModelBuilder.resnet")
except ValueError:pass
else:raise AssertionError("development wrapper leaked into installed help")
assert show_topic("components.fusion","de")["lang"]=="de"
from hyso.cli import main
assert main(["docs","show","models.graph-builder","--lang","tr","--json"])==0
'''
        code='INVENTORY='+repr(INVENTORY)+'\nFUSIONS='+repr(FUSIONS)+'\n'+code
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8');env.pop('PYTHONPATH',None)
        probe=work/(kind+'-registry-probe.py')
        probe.write_text(code,encoding='utf-8')
        p=subprocess.run([sys.executable,'-I','-B',str(probe),str(target)],cwd=work,env=env,capture_output=True,text=True,encoding='utf-8',timeout=120)
        assert p.returncode==0,p.stdout+p.stderr
        jedi=pytest.importorskip('jedi')
        paths=[str(target),*[p for p in sys.path if p and not Path(p).resolve().is_relative_to(ROOT)]]
        project=jedi.Project(str(work),sys_path=paths)
        prefix='from hyso import ModelBuilder, GraphModelBuilder\nb=ModelBuilder((16,8,8),3)\ng=GraphModelBuilder({"image":[16,8,8]})\n'
        for variable,key in (('b','sequential'),('g','graph')):
            names={c.name for c in jedi.Script(prefix+variable+'.',path=str(work/'completion_probe.py'),project=project).complete()}
            expected={n for n,row in INVENTORY.items() if row[key] and 'commercial' in row['profiles']}
            forbidden={n for n,row in INVENTORY.items() if not row[key] or 'commercial' not in row['profiles']}
            assert expected.issubset(names),expected-names
            assert names.isdisjoint(forbidden),names&forbidden
