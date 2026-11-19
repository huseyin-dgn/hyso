"""UX-06: exact metadata inference, native validation and CPU compatibility."""
import ast
import inspect
import json
from pathlib import Path
import runpy
import subprocess
import typing

import pytest
import torch

from hyso import GraphModelBuilder, ModelBuilder, ModelConfigError, ModelRegistry

ROOT=Path(__file__).resolve().parents[1]
BASELINE='fe728fb6f630aaf56346050b4e8ac06e2ccf0cdf'


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    torch.set_num_threads(1)
    def forbidden(*args,**kwargs):pytest.fail('smart builder attempted GPU access or training')
    for name in ('init','_lazy_init','is_available','device_count','synchronize'):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    from hyso import Trainer,TrainingSession
    monkeypatch.setattr(Trainer,'fit',forbidden)
    monkeypatch.setattr(TrainingSession,'fit',forbidden)


def sequential(schema=2,pooling='avg',short=True):
    b=ModelBuilder.classification((3,8,8),num_classes=2,pooling=pooling) if schema==2 else ModelBuilder((3,8,8),2)
    b.conv2d(out_channels=4,**({} if short else {'in_channels':3})).batch_norm(**({} if short else {'num_features':4})).relu().max_pool2d(kernel_size=2)
    return b.conv2d(out_channels=6,**({} if short else {'in_channels':4})).batch_norm(**({} if short else {'num_features':6})).relu()


def graph(pooling='avg',short=True):
    b=GraphModelBuilder({'image':[3,8,8]})
    b.conv2d('stem',['image'],out_channels=4,**({} if short else {'in_channels':3}))
    b.batch_norm('norm',['stem'],**({} if short else {'num_features':4})).relu('act',['norm']).max_pool2d('pool',['act'],kernel_size=2)
    b.conv2d('next',['pool'],out_channels=6,**({} if short else {'in_channels':4})).batch_norm('next_norm',['next'],**({} if short else {'num_features':6}))
    features=96 if pooling=='none' else 6
    return b.classification_head('head',['next_norm'],num_classes=2,pooling=pooling,**({} if short else {'in_features':features})).outputs(logits='head')


@pytest.mark.parametrize('kind,pooling',[('schema1','avg'),('schema2','avg'),('schema2','max'),('schema2','none'),('schema3','avg'),('schema3','max'),('schema3','none')])
def test_explicit_short_json_state_dict_and_cpu_output_equivalence(kind,pooling,tmp_path):
    make=(lambda short:graph(pooling,short)) if kind=='schema3' else (lambda short:sequential(1 if kind=='schema1' else 2,pooling,short))
    short,explicit=make(True),make(False)
    assert short.to_config()==explicit.to_config()
    path=tmp_path/'model.json';short.to_json(path)
    restored=type(short).from_json(path)
    assert restored.to_config()==explicit.to_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17);a=short.build().eval()
        torch.manual_seed(17);b=explicit.build().eval()
        torch.manual_seed(17);c=restored.build().eval()
    assert a.state_dict().keys()==b.state_dict().keys()==c.state_dict().keys()
    for name,value in a.state_dict().items():
        assert torch.equal(value,b.state_dict()[name]) and torch.equal(value,c.state_dict()[name])
    with torch.inference_mode():
        inputs=torch.ones(2,3,8,8)
        first=a(inputs);second=b(inputs);third=c(inputs)
        if kind=='schema3':
            assert first.keys()==second.keys()==third.keys()=={'logits'}
            first,second,third=first['logits'],second['logits'],third['logits']
    assert first.shape==second.shape==third.shape==(2,2)
    assert torch.equal(first,second) and torch.equal(first,third)
    assert json.loads(path.read_text(encoding='utf-8'))==explicit.to_config()


@pytest.mark.parametrize('pooling,features',[('avg',6),('max',6),('none',96)])
def test_classification_keeps_existing_schema2_head_inference(pooling,features):
    b=sequential(pooling=pooling)
    config=b.to_config();report=b.preflight(config)
    assert config['schema_version']==2 and config['kind']=='sequential'
    assert 'in_features' not in config['head']['params']
    assert report['valid'] and report['head']['in_features']==features
    assert b.build().head.linear.in_features==features


def test_materialized_channels_continue_after_json_and_channel_rearrangement(tmp_path):
    b=ModelBuilder.classification((16,4,4),num_classes=2).pixel_shuffle(upscale_factor=2).batch_norm().conv2d(out_channels=8)
    assert b.to_config()['blocks'][1]['params']['num_features']==4
    assert b.to_config()['blocks'][2]['params']['in_channels']==4
    path=tmp_path/'model.json';b.to_json(path)
    restored=ModelBuilder.from_json(path).batch_norm().conv2d(out_channels=2)
    assert restored.to_config()['blocks'][-2]['params']['num_features']==8
    assert restored.to_config()['blocks'][-1]['params']['in_channels']==8


@pytest.mark.parametrize('method,parameter',[('conv2d','in_channels'),('batch_norm','num_features')])
@pytest.mark.parametrize('value',[False,True,0,-1,1.5,'3',4])
@pytest.mark.parametrize('cls',[ModelBuilder,GraphModelBuilder])
def test_explicit_invalid_or_mismatching_values_are_not_corrected(cls,method,parameter,value):
    b=cls({'image':[3,8,8]}) if cls is GraphModelBuilder else cls((3,8,8),2)
    kwargs={parameter:value,**({'out_channels':4} if method=='conv2d' else {})};before=b.to_config()
    with pytest.raises(ModelConfigError):
        getattr(b,method)(*(['node',['image']] if cls is GraphModelBuilder else []),**kwargs)
    assert b.to_config()==before


@pytest.mark.parametrize('cls',[ModelBuilder,GraphModelBuilder])
def test_none_explicitly_requests_inference_and_group_validation_remains_native(cls):
    b=cls({'image':[3,8,8]}) if cls is GraphModelBuilder else cls((3,8,8),2)
    prefix=['node',['image']] if cls is GraphModelBuilder else []
    before=b.to_config()
    with pytest.raises(ModelConfigError):b.conv2d(*prefix,out_channels=4,groups=2)
    assert b.to_config()==before
    b.conv2d(*prefix,in_channels=None,out_channels=4)
    params=b.to_config()['nodes'][0]['params'] if cls is GraphModelBuilder else b.to_config()['blocks'][0]['params']
    assert type(params['in_channels']) is int and params['in_channels']==3


def test_flattened_sequential_features_and_fixed_profiles_require_supported_shapes():
    b=ModelBuilder.classification((3,4,4),num_classes=2,pooling='none').flatten()
    for operation in (lambda:b.conv2d(out_channels=4),lambda:b.batch_norm()):
        before=b.to_config()
        with pytest.raises(ModelConfigError,match='Cannot infer') as error:operation()
        assert error.value.code=='SHAPE_INFERENCE_UNAVAILABLE' and b.to_config()==before
    fixed=ModelBuilder.modular_cnn(input_channels=3,num_classes=2,widths=(4,),repeats=(1,),strides=(1,))
    with pytest.raises(ModelConfigError) as error:fixed.batch_norm()
    assert error.value.code=='INVALID_BLOCK_ORDER'
    with pytest.raises(ModelConfigError):ModelBuilder((None,4,4),2)


@pytest.mark.parametrize('fusion,expected',[('concat',6),('add',3),('multiply',3),('weighted_sum',3)])
def test_graph_infers_from_verified_fusion_result(fusion,expected):
    b=GraphModelBuilder({'image':[3,4,4]}).conv2d('a',['image'],out_channels=3).conv2d('b',['image'],out_channels=3)
    getattr(b,'add_inputs' if fusion=='add' else fusion)('merge',['a','b'],**({'weights':[.25,.75]} if fusion=='weighted_sum' else {}))
    b.batch_norm('norm',['merge']).classification_head('head',['norm'],num_classes=2).outputs(logits='head')
    rows={r['id']:r for r in b.to_config()['nodes']}
    assert rows['norm']['params']['num_features']==rows['head']['params']['in_features']==expected
    assert b.preview()['valid']


@pytest.mark.parametrize('case',['forward','cycle','multiple','bad_source','bad_fusion'])
def test_ambiguous_invalid_graph_sources_require_explicit_sizes_without_mutation(case):
    b=GraphModelBuilder({'image':[3,4,4]});source=['missing']
    if case=='cycle':
        b.add('a','identity',['b']).add('b','identity',['a']);source=['a']
    elif case=='multiple':source=['image','image']
    elif case=='bad_source':
        b.add('bad','conv2d',['image'],in_channels=9,out_channels=4);source=['bad']
    elif case=='bad_fusion':
        b.conv2d('a',['image'],out_channels=3).conv2d('b',['image'],out_channels=4).add_inputs('merge',['a','b']);source=['merge']
    before=b.to_config()
    with pytest.raises(ModelConfigError,match='Cannot infer') as error:b.batch_norm('new',source)
    assert error.value.code=='SHAPE_INFERENCE_UNAVAILABLE' and b.to_config()==before


def test_explicit_forward_reference_and_unrelated_partial_branches_are_retained():
    b=GraphModelBuilder({'image':[3,4,4]})
    b.conv2d('early',['late'],in_channels=3,out_channels=4)
    b.conv2d('late',['image'],out_channels=3)
    b.classification_head('head',['early'],num_classes=2).outputs(logits='head')
    assert b.preview()['valid']


def test_graph_uses_the_named_source_instead_of_the_last_node():
    b=GraphModelBuilder({'image':[3,4,4]}).conv2d('left',['image'],out_channels=4).conv2d('right',['image'],out_channels=7)
    b.batch_norm('norm',['left']).classification_head('head',['norm'],num_classes=2).outputs(logits='head',features='right')
    rows={r['id']:r for r in b.to_config()['nodes']}
    assert rows['norm']['params']['num_features']==rows['head']['params']['in_features']==4
    assert b.preview()['valid']


@pytest.mark.parametrize('cls',[ModelBuilder,GraphModelBuilder])
def test_invalid_spatial_geometry_is_not_repaired(cls):
    b=cls({'image':[3,4,4]}) if cls is GraphModelBuilder else cls((3,4,4),2)
    before=b.to_config()
    with pytest.raises(ModelConfigError):
        b.conv2d(*(['node',['image']] if cls is GraphModelBuilder else []),out_channels=4,kernel_size=9,padding=0)
    assert b.to_config()==before


def test_nonsquare_flattened_classification_head_uses_the_full_feature_product():
    b=ModelBuilder.classification((2,5,7),num_classes=3,pooling='none').conv2d(out_channels=4).batch_norm()
    assert b.preflight(b.to_config())['head']['in_features']==4*5*7


def test_graph_vector_head_requires_explicit_compatible_pooling_and_dimensions():
    b=GraphModelBuilder({'vector':[12]});before=b.to_config()
    with pytest.raises(ModelConfigError):b.classification_head('head',['vector'],num_classes=2)
    assert b.to_config()==before
    with pytest.raises(ModelConfigError,match='Cannot infer'):b.conv2d('conv',['vector'],out_channels=4)
    with pytest.raises(ModelConfigError):b.classification_head('head',['vector'],num_classes=2,pooling='none',in_features=7)
    assert b.to_config()==before
    b.classification_head('head',['vector'],num_classes=2,pooling='none').outputs(logits='head')
    assert b.to_config()['nodes'][0]['params']['in_features']==12 and b.preview()['valid']


def test_inference_is_metadata_only(monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('inference allocated a module, loaded weights, or ran forward')
    monkeypatch.setattr(torch.nn.Module,'__init__',forbidden)
    monkeypatch.setattr(torch.nn.Module,'_call_impl',forbidden)
    monkeypatch.setattr(torch,'load',forbidden)
    assert sequential().to_config()['blocks'][-2]['params']['num_features']==6
    assert graph().preview()['valid']


def test_real_signatures_jedi_and_inference_metadata():
    import jedi
    expected=[(ModelBuilder,'conv2d','in_channels','sequential'),(ModelBuilder,'batch_norm','num_features','sequential'),
              (GraphModelBuilder,'conv2d','in_channels','graph'),(GraphModelBuilder,'batch_norm','num_features','graph'),
              (GraphModelBuilder,'classification_head','in_features','graph')]
    metadata=json.loads((ROOT/'hyso/data/api-topics.json').read_text(encoding='utf-8'))['registry_completion']
    for cls,name,param,mode in expected:
        method=getattr(cls,name);sig=inspect.signature(method)
        assert sig.parameters[param].default is None
        assert typing.get_type_hints(method)[param]==int | None
        assert not any(p.kind in (p.VAR_KEYWORD,p.VAR_POSITIONAL) for p in sig.parameters.values())
        assert metadata['components'][name][mode]['inferred_parameters'][param]['default'] is None
    source='from hyso import ModelBuilder\nb=ModelBuilder.classification((3,8,8),num_classes=2)\nb.conv2d('
    signatures=jedi.Script(source).get_signatures(3,len('b.conv2d('))
    assert signatures and {'in_channels','out_channels','kernel_size'}<={p.name for p in signatures[0].params}
    source='from hyso import GraphModelBuilder\nb=GraphModelBuilder({"image":[3,8,8]})\nb.classification_head('
    signatures=jedi.Script(source).get_signatures(3,len('b.classification_head('))
    assert signatures and {'node_id','inputs','in_features','num_classes','pooling'}<={p.name for p in signatures[0].params}
    assert len(metadata['components'])==len(ModelRegistry().list())==85
    assert sum('commercial' in c['profiles'] for c in metadata['components'].values())==57 and len(metadata['fusions'])==4


def test_native_geometry_algorithms_are_unchanged_from_ux05():
    files={'hyso/models/builder.py':{'_shape','_validate_head','_validate'},
           'hyso/models/graph.py':{'_inputs','_structure','_describe','_analyze'},
           'hyso/models/sampling.py':{'head_geometry'},'hyso/models/graph_ops.py':{'parameters','fusion_shape'}}
    for filename,names in files.items():
        old=subprocess.run(['git','show',BASELINE+':'+filename],cwd=ROOT,capture_output=True)
        if old.returncode:pytest.skip('UX-06 native geometry audit requires UX-05 history')
        original={n.name:ast.dump(n) for n in ast.walk(ast.parse(old.stdout.decode('utf-8'))) if isinstance(n,ast.FunctionDef) and n.name in names}
        current={n.name:ast.dump(n) for n in ast.walk(ast.parse((ROOT/filename).read_text(encoding='utf-8'))) if isinstance(n,ast.FunctionDef) and n.name in names}
        assert current==original


def test_short_cpu_example(tmp_path):
    module=runpy.run_path(str(ROOT/'examples/models/smart_builder.py'))
    b=module['cnn_builder']()
    assert b.to_config()['schema_version']==2
    with torch.inference_mode():assert b.build().eval()(torch.ones(2,1,8,8)).shape==(2,3)
