"""UX-05 focused CPU compatibility, signatures and static discovery checks."""
import ast
import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import runpy
import subprocess
import typing

import pytest
import torch

from hyso import Device, ModelBuilder, GraphModelBuilder, TrainingConfig, TrainingConfigError
from hyso import analysis
from hyso.models import ModelRegistry

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '7e954d1a3d2bf1bfd2d9afe44500a3c3e2d2ece5'


def require_audit_history():
    for revision in (BASELINE,'8ed5c0f'):
        result=subprocess.run(['git','cat-file','-e',revision+':hyso/data/api-topics.json'],
                              cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if result.returncode:
            pytest.skip('UX-05 historical audit requires the pre-UX-05 source commits')


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('UX-05 CPU tests attempted a CUDA query or initialization')
    for name in ('_lazy_init', 'init', 'is_available', 'device_count', 'synchronize'):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.mark.parametrize('index', [0, 1, 17])
def test_device_selectors_are_plain_strings_without_hardware_access(index):
    assert type(Device.cpu()) is str and Device.cpu() == 'cpu'
    assert type(Device.cuda(index)) is str and Device.cuda(index) == f'cuda:{index}'
    assert json.loads(json.dumps({'device': Device.cuda(index)})) == {'device': f'cuda:{index}'}


@pytest.mark.parametrize('index', [True, False, -1, 1.5, '0', None])
def test_device_rejects_noninteger_or_negative_indices(index):
    with pytest.raises(ValueError, match='nonnegative integer'):
        Device.cuda(index)


TRAINING_CASES = [
    ('with_sgd', {'momentum': .9, 'nesterov': True}, {'optimizer': 'sgd', 'optimizer_params': dict(momentum=.9, dampening=0, weight_decay=0, nesterov=True)}),
    ('with_adam', {'eps': 1e-7}, {'optimizer': 'adam', 'optimizer_params': dict(betas=(.9, .999), eps=1e-7, weight_decay=0)}),
    ('with_adamw', {'weight_decay': .01}, {'optimizer': 'adamw', 'optimizer_params': dict(betas=(.9, .999), eps=1e-8, weight_decay=.01)}),
    ('with_step_scheduler', {'step_size': 2}, {'scheduler': 'step', 'scheduler_params': dict(step_size=2, gamma=.1)}),
    ('with_cosine_scheduler', {}, {'scheduler': 'cosine', 'scheduler_params': dict(eta_min=0)}),
    ('with_onecycle_scheduler', {'max_lr': .01}, {'scheduler': 'onecycle', 'scheduler_params': dict(max_lr=.01, pct_start=.3, div_factor=25, final_div_factor=1e4, cycle_momentum=False)}),
    ('with_plateau_scheduler', {'patience': 3}, {'scheduler': 'plateau', 'scheduler_params': dict(factor=.1, patience=3, threshold=1e-4, cooldown=0, min_lr=0, eps=1e-8)}),
    ('with_early_stopping', {'patience': 3}, {'early_stopping': dict(patience=3, delta=0, restore_best_weights=True)}),
]


@pytest.mark.parametrize('name,kwargs,legacy', TRAINING_CASES)
def test_training_convenience_preserves_schema_defaults_and_immutability(name, kwargs, legacy, tmp_path):
    base = TrainingConfig(epochs=4, validation=True, learning_rate=.02)
    before = base.to_dict()
    new = getattr(base, name)(**kwargs)
    old = TrainingConfig(epochs=4, validation=True, learning_rate=.02, **legacy)
    assert new.to_dict() == old.to_dict()
    assert base.to_dict() == before and new is not base
    path = tmp_path / 'config.json'
    new.to_json(path)
    assert TrainingConfig.from_json(path).to_dict() == old.to_dict()
    assert new.to_dict()['learning_rate'] == .02


@pytest.mark.parametrize('name,kwargs,legacy', [
    ('with_sgd', {'nesterov': True}, dict(optimizer='sgd', optimizer_params=dict(momentum=0, dampening=0, weight_decay=0, nesterov=True))),
    ('with_adam', {'betas': (1, .9)}, dict(optimizer='adam', optimizer_params=dict(betas=(1,.9), eps=1e-8, weight_decay=0))),
    ('with_step_scheduler', {'step_size': 0}, dict(scheduler='step', scheduler_params=dict(step_size=0,gamma=.1))),
    ('with_onecycle_scheduler', {'max_lr': -1}, dict(scheduler='onecycle', scheduler_params=dict(max_lr=-1,pct_start=.3,div_factor=25,final_div_factor=1e4,cycle_momentum=False))),
    ('with_plateau_scheduler', {}, dict(scheduler='plateau',scheduler_params=dict(factor=.1,patience=10,threshold=1e-4,cooldown=0,min_lr=0,eps=1e-8))),
    ('with_early_stopping', {}, dict(early_stopping=dict(patience=7,delta=0,restore_best_weights=True))),
])
def test_training_convenience_uses_original_validator(name, kwargs, legacy):
    base = TrainingConfig()
    before = base.to_dict()
    with pytest.raises(TrainingConfigError) as old:
        TrainingConfig(**legacy)
    with pytest.raises(TrainingConfigError) as new:
        getattr(base, name)(**kwargs)
    assert str(new.value) == str(old.value)
    assert base.to_dict() == before


@pytest.mark.parametrize('method,params,head', [
    ('classification', dict(num_classes=2, dropout=.1), 'classification'),
    ('regression', dict(output_features=3), 'regression'),
    ('embedding', dict(embedding_dim=4, normalize=True), 'embedding'),
])
def test_explicit_sequential_heads_match_legacy_schema2(method, params, head):
    new = getattr(ModelBuilder, method)((2, 3, 4), **params)
    assert new.to_config() == ModelBuilder.sequential((2,3,4), head=head, **params).to_config()
    assert new.build()(torch.ones(2,2,3,4)).shape[0] == 2
    with pytest.raises(ValueError):
        getattr(ModelBuilder, method)((2,3,4), pooling='invalid', **params)


def test_registry_selectors_follow_both_profiles():
    import hyso.models.registry as registry
    assert set(typing.get_type_hints(registry.ModelRegistry.get)['name'].__args__) == {s.name for s in ModelRegistry().list()}
    tree = ast.parse((ROOT/'setup.py').read_text(encoding='utf-8'))
    helper = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='commercial_payload')
    scope = {'json':json}
    exec(compile(ast.Module(body=[helper],type_ignores=[]),'setup.py','exec'),scope)
    name = 'hyso/models/registry.py'
    data = scope['commercial_payload'](name,(ROOT/name).read_bytes())
    assert scope['commercial_payload'](name,data) == data
    projected = ast.parse(data)
    alias = next(n.value for n in projected.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='_ComponentName' for a in n.targets))
    names = ast.literal_eval(alias.slice)
    index=json.loads((ROOT/'hyso/data/api-topics.json').read_text(encoding='utf-8'))['registry_completion']['components']
    assert set(names)=={k for k,v in index.items() if 'commercial' in v['profiles']} and len(names)==57


REGISTRY = ModelRegistry().list()
NATIVE_CONSTRUCTORS = tuple(c for c in REGISTRY if c.constructor.__module__ in ('hyso.models.advanced','hyso.models.sampling')
                            and c.name in ('bottleneck_residual','preactivation_residual','wide_residual_block','grouped_bottleneck','inverted_bottleneck','gated_residual','spatial_self_attention','multihead_spatial_attention','classification_head','regression_head','embedding_head'))


@pytest.mark.parametrize('spec',NATIVE_CONSTRUCTORS,ids=lambda s:s.name)
def test_direct_native_constructors_have_exact_registry_signatures_and_weights(spec):
    signature=inspect.signature(spec.constructor)
    assert all(p.kind != p.VAR_KEYWORD for p in signature.parameters.values())
    assert set(signature.parameters)=={p.name for p in spec.parameters}
    params={p.name:4 for p in spec.parameters if p.required}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19); direct=spec.constructor(**params)
        torch.manual_seed(19); registered=ModelRegistry().create(spec.name,**params)
    assert direct.state_dict().keys()==registered.state_dict().keys()
    assert all(torch.equal(v,registered.state_dict()[k]) for k,v in direct.state_dict().items())


def test_direct_constructor_literal_choices_match_authoritative_registry():
    for spec in REGISTRY:
        if not spec.constructor.__module__.startswith('hyso.'):
            continue
        hints=typing.get_type_hints(spec.constructor.__init__)
        for parameter in spec.parameters:
            if parameter.choices and any(isinstance(v,str) for v in parameter.choices):
                assert typing.get_origin(hints[parameter.name]) is typing.Literal,(spec.name,parameter.name)
                assert set(typing.get_args(hints[parameter.name]))==set(parameter.choices),(spec.name,parameter.name)


def test_analysis_all_35_ids_and_pure_planning(monkeypatch):
    from hyso.analysis import execution
    def forbidden(*args,**kwargs):pytest.fail('planning performed a workload')
    monkeypatch.setattr(execution,'_invoke',forbidden)
    monkeypatch.setattr(torch.nn.Module,'__init__',forbidden)
    hints=typing.get_type_hints(analysis.plan_analysis)
    literal=next(t for t in typing.get_args(hints['features']) if typing.get_origin(t) is typing.Literal)
    assert set(typing.get_args(literal))=={f'P{i:02d}' for i in range(1,36)}
    for feature in typing.get_args(literal):
        assert analysis.plan_analysis(analysis.ModelContext('user-model'),features=feature).to_dict()['selected_features']==[feature]
    assert TrainingConfig(device=Device.cpu()).with_adamw().to_dict()['device']=='cpu'


def test_graph_head_spec_is_plain_data_and_native_validation_is_retained():
    from hyso.analysis.graph_heads import _spec
    spec=analysis.GraphHeadSpec(target='label',task='multiclass',class_names=['a','b'],split='custom-split',dataset_id='my-data',preprocessing_id='v1')
    assert type(spec) is dict and _spec(spec)['zero_division']==0
    assert analysis.GraphHeadSpec.__required_keys__=={'target','task','class_names','split','dataset_id','preprocessing_id'}
    with pytest.raises(NotImplementedError):_spec(dict(spec,task='invented'))


def test_shared_feature_metric_help_and_precision_selectors_are_typed():
    import hyso_guide
    from hyso.advisor import Metric, NativeMetric, NativeMetricResolver, goal_context
    from hyso.performance import run_sweep
    expected={f'P{i:02d}' for i in range(1,36)}
    for method in (analysis.EvidenceRecord.from_json,analysis.EvidenceRecord.unavailable):
        assert set(typing.get_args(typing.get_type_hints(method)['feature_id']))==expected
    assert typing.get_type_hints(goal_context)['metric'] is Metric
    assert typing.get_type_hints(NativeMetricResolver.resolve)['metric'] is NativeMetric
    assert typing.get_type_hints(NativeMetricResolver.check_reference)['metric'] is NativeMetric
    hints=typing.get_type_hints(hyso_guide.show_topic)
    assert set(typing.get_args(hints['topic_id']))=={t['id'] for t in hyso_guide.list_topics('en')}
    points=typing.get_args(typing.get_type_hints(run_sweep)['points'])[0]
    precision=next(t for t in typing.get_args(points) if typing.get_origin(t) is typing.Literal)
    assert set(typing.get_args(precision))=={'fp32','fp64','amp_fp16','bf16','int8_dynamic'}


def test_opened_wrappers_forward_native_parameters(monkeypatch):
    from hyso.performance import pipeline
    calls=[]; sentinel=object()
    monkeypatch.setattr(pipeline,'measure_pipeline',lambda model,source,**kwargs:calls.append((model,source,kwargs)) or sentinel)
    monkeypatch.setattr(pipeline,'analyze_stages',lambda result:result)
    policy=pipeline.PipelineProtocol()
    assert pipeline.measure_stages('model','source',protocol=policy) is sentinel
    assert calls==[('model','source',dict(preprocess=None,transfer=None,postprocess=None,deliver=None,protocol=policy))]
    builder=GraphModelBuilder({'image':[1,2,2]}).classification_head('head',['image'],in_features=1,num_classes=2).outputs(logits='head')
    assert builder.preview(batch_size=2)==builder.preflight(builder.to_config(),batch_size=2)


def test_jedi_discovers_real_methods_and_named_parameters():
    import jedi
    script=jedi.Script('from hyso import TrainingConfig\nc=TrainingConfig()\nc.with_')
    assert {name for name,_,_ in TRAINING_CASES} <= {c.name for c in script.complete(3,len('c.with_'))}
    code='from hyso import TrainingConfig\nTrainingConfig().with_onecycle_scheduler('
    signatures=jedi.Script(code).get_signatures(2,len(code.splitlines()[1]))
    assert signatures and {p.name for p in signatures[0].params}>={'max_lr','pct_start','cycle_momentum'}
    code='from hyso import Device\nDevice.'
    assert {'cpu','cuda'} <= {c.name for c in jedi.Script(code).complete(2,len('Device.'))}
    from hyso.models.fit import CNNSENormalWithFit
    assert not any(p.kind in (p.VAR_POSITIONAL,p.VAR_KEYWORD) for p in inspect.signature(CNNSENormalWithFit).parameters.values())
    assert 'GraphHeadSpec' in dir(analysis)


def test_complete_original_api_and_alias_inventory(tmp_path):
    require_audit_history()
    import sys
    subprocess.run([sys.executable,str(ROOT/'tools/docs/audit_pythonic_api.py'),'--output',str(tmp_path)],
                   cwd=ROOT,check=True,capture_output=True)
    summary=json.loads((tmp_path/'audit-summary.json').read_text(encoding='utf-8'))
    assert summary['original_ux03_declarations']==775
    assert summary['current_indexed_declarations']>=956
    assert not summary['original_ux03_missing'] and not summary['original_aliases_missing'] and not summary['indexed_source_missing']


def test_existing_native_function_bodies_are_preserved():
    """Annotation edits and explicit constructors must not rewrite native engines."""
    require_audit_history()
    class Normalize(ast.NodeTransformer):
        def visit_arg(self, node):
            node.annotation = None
            return node

        def visit_FunctionDef(self, node):
            node.returns = None
            if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
                node.body=node.body[1:]
            return self.generic_visit(node)

    def functions(tree):
        result={}
        def walk(nodes,prefix=''):
            for n in nodes:
                if isinstance(n,ast.ClassDef):walk(n.body,prefix+n.name+'.')
                elif isinstance(n,ast.If):walk(n.body,prefix);walk(n.orelse,prefix)
                elif isinstance(n,ast.FunctionDef):result[prefix+n.name]=n
        walk(tree.body)
        return result

    changed=subprocess.check_output(['git','diff','--name-only',BASELINE,'--','hyso','hyso_guide.py'],cwd=ROOT,text=True).splitlines()
    # Later DE/ES presentation work has its own reviewed baseline. Keep checking
    # those exact bodies instead of comparing them to pre-localization UX-05.
    localized=json.loads((ROOT/'tests/fixtures/localized-presentation-body-baseline-v1.json').read_text(encoding='utf-8'))
    presentation_bodies={(row['path'],row['function']):row['normalized_body_sha256'] for row in localized['functions']}
    checked=0
    exceptions={('hyso/models/graph.py','GraphModelBuilder.preview'),('hyso/performance/pipeline.py','measure_stages'),
                ('hyso/analysis/__init__.py','__getattr__'),('hyso/models/fit.py','CNNSENormalWithFit.__init__')}
    # UX-06 adds metadata-only inference in these delegates; native geometry stays protected.
    exceptions.update({('hyso/models/builder.py','ModelBuilder.conv2d'),
                       ('hyso/models/builder.py','ModelBuilder.batch_norm'),
                       ('hyso/models/graph.py','GraphModelBuilder.conv2d'),
                       ('hyso/models/graph.py','GraphModelBuilder.batch_norm'),
                       ('hyso/models/graph.py','GraphModelBuilder.classification_head')})
    engines={'SpatialSelfAttention','AdvancedResidual','PredictionHead'}
    for filename in changed:
        if not filename.endswith('.py'):continue
        original=functions(ast.parse(subprocess.check_output(['git','show',f'{BASELINE}:{filename}'],cwd=ROOT).decode('utf-8')))
        current=functions(ast.parse((ROOT/filename).read_text(encoding='utf-8')))
        for name,old in original.items():
            if (filename,name) in exceptions:continue
            target=name
            if filename in ('hyso/models/advanced.py','hyso/models/sampling.py') and name.endswith('.__init__') and name.split('.')[0] in engines:
                target=name.replace('.__init__','._initialize')
            assert target in current,(filename,target)
            before=Normalize().visit(copy.deepcopy(old));after=Normalize().visit(copy.deepcopy(current[target]))
            current_body=ast.dump(ast.Module(body=after.body,type_ignores=[]))
            if (filename,name) in presentation_bodies:
                assert hashlib.sha256(current_body.encode()).hexdigest()==presentation_bodies[(filename,name)],(filename,name)
            else:
                assert ast.dump(ast.Module(body=before.body,type_ignores=[]))==current_body,(filename,name)
            checked+=1
    assert checked>200


@pytest.mark.parametrize('execute',[False,True])
def test_cpu_example_prepare_and_explicit_execution(execute,tmp_path):
    api=runpy.run_path(str(ROOT/'examples/pythonic_api.py'))
    result=api['run'](execute=execute,output=tmp_path)
    assert result['executed'] is execute
    if execute:
        assert {r['status'] for r in result['analysis']['features']}=={'COMPLETED'}
        assert {p.name for p in tmp_path.iterdir()}=={'report.json','report.md','report.html'}
    else:
        assert not list(tmp_path.iterdir())
