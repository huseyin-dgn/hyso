"""Public help/type contracts tied to the real closed component schemas."""
import ast
import inspect
import re
from pathlib import Path

import pytest

from tools.api_documentation_audit import public_definitions, parameters

ROOT = Path(__file__).resolve().parents[1]


def test_public_parameter_sections_cover_every_explicit_argument():
    missing=[]
    for filename, name, node in public_definitions():
        doc=ast.get_docstring(node) or ''
        if not doc:
            missing.append((filename,name,'docstring'))
        for param in parameters(node):
            if not re.search(r'(?m)^\s*\*{0,2}'+re.escape(param)+r'\s*(?:\(|:)',doc):
                missing.append((filename,name,param))
    assert not missing, missing


@pytest.mark.parametrize('filename,owner,operation',[
    ('builder.py','ModelBuilder','add'),('graph.py','GraphModelBuilder','add'),
    ('registry.py','ModelRegistry','create')])
def test_literal_overloads_match_runtime_parameter_schemas(filename,owner,operation):
    from hyso.models.registry import ModelRegistry
    tree=ast.parse((ROOT/'hyso/models'/filename).read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner)
    overloads=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==operation and
               any(isinstance(d,ast.Attribute) and d.attr=='overload' for d in n.decorator_list)]
    registry=ModelRegistry()
    for node in overloads:
        selector=next(a for a in node.args.args if a.arg in ('name','component'))
        component=ast.literal_eval(selector.annotation.slice)
        if component in ('add','concat','multiply','weighted_sum'):
            continue
        fields=registry.get(component).parameters
        assert [a.arg for a in node.args.kwonlyargs]==[p.name for p in fields]
        assert [v is None for v in node.args.kw_defaults]==[p.required for p in fields]
        assert all(a.annotation is not None for a in node.args.kwonlyargs)
        for field,default in zip(fields,node.args.kw_defaults):
            if not field.required:
                value=ast.literal_eval(default)
                expected=list(field.default) if isinstance(field.default,tuple) else field.default
                assert value==expected
    expected={'ModelBuilder':53,'GraphModelBuilder':60,'ModelRegistry':57}
    assert len(overloads)==expected[owner]


def test_builder_inference_help_and_return_annotations():
    from hyso import ModelBuilder,GraphModelBuilder
    from hyso.models.graph import GraphModel
    assert inspect.signature(ModelBuilder.conv2d).parameters['in_channels'].default is None
    assert 'None infers C' in inspect.getdoc(ModelBuilder.conv2d)
    assert 'None infers C' in inspect.getdoc(GraphModelBuilder.conv2d)
    assert 'dict[str, torch.Tensor]' in str(inspect.signature(GraphModel.forward).return_annotation)
    assert 'output_padding' in inspect.getdoc(ModelBuilder.conv_transpose2d)
    assert 'dilation' in inspect.getdoc(ModelBuilder.conv2d)


def test_training_keyword_types_describe_supported_options():
    from hyso.training import config
    assert set(config._OptimizerOptions.__annotations__)=={
        'momentum','dampening','weight_decay','nesterov','betas','eps'}
    assert {'step_size','gamma','t_max','max_lr','patience'}<=set(config._SchedulerOptions.__annotations__)
    assert not config._OptimizerOptions.__total__


def test_parameter_docs_use_the_google_sections_recognized_by_pylance():
    from hyso import ModelBuilder,GraphModelBuilder
    for cls in (ModelBuilder,GraphModelBuilder):
        assert '\nArgs:\n' in inspect.getdoc(cls.conv2d)
        tree=ast.parse(inspect.getsource(cls))
        nodes=[n for n in tree.body[0].body if isinstance(n,ast.FunctionDef) and n.name=='add' and
               any(isinstance(d,ast.Attribute) and d.attr=='overload' for d in n.decorator_list)]
        conv=next(n for n in nodes if ast.literal_eval(next(a.annotation.slice for a in n.args.args if a.arg in ('name','component')))=='conv2d')
        assert 'kernel_size:' in ast.get_docstring(conv)
        assert 'dilation:' in ast.get_docstring(conv)


def test_documented_callback_lifecycle_remains_explicit(tmp_path):
    import torch
    from hyso.callbacks.callbacks import EarlyStopping,ModelCheckpoint
    model=torch.nn.Linear(2,1)
    stop=EarlyStopping(patience=2,verbose=False)
    stop(1.0,model);stop(1.1,model);assert not stop.early_stop
    stop(1.1,model);assert stop.early_stop
    dest=tmp_path/'weights.pt';checkpoint=ModelCheckpoint(str(dest),verbose=False)
    checkpoint(0,model,1.0);before=dest.read_bytes()
    checkpoint(1,model,1.1);assert dest.read_bytes()==before
