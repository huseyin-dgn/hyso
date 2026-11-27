"""Offline help and hover metadata; no model construction or measurement workloads."""
import ast
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import typing
from urllib.parse import unquote, urlsplit
import venv
import webbrowser

import pytest
from test_packaging import distributions

import hyso_guide

ROOT=Path(__file__).resolve().parents[1]
INDEX=json.loads((ROOT/'hyso/data/api-topics.json').read_text(encoding='utf-8'))
TOPICS=json.loads((ROOT/'hyso/data/user-guide-topics.json').read_text(encoding='utf-8'))
CONTENT=json.loads((ROOT/'docs/user-guide/content.json').read_text(encoding='utf-8'))


@pytest.fixture(autouse=True)
def no_workloads(monkeypatch):
    import torch
    def forbidden(*args,**kwargs):pytest.fail('contextual help attempted a workload/device query')
    monkeypatch.setattr(torch.nn.Module,'__init__',forbidden)
    monkeypatch.setattr(torch.nn.Module,'_call_impl',forbidden)
    monkeypatch.setattr(torch,'load',forbidden)
    for name in ('init','_lazy_init','is_available','device_count','synchronize'):
        monkeypatch.setattr(torch.cuda,name,forbidden)


@pytest.mark.parametrize('lang',['tr','en','de','es'])
def test_every_localized_topic_reads_the_same_html_with_exact_fragment(lang):
    rows=hyso_guide.list_topics(lang)
    assert len(rows)==len(CONTENT['topics'])>=70
    for row,source in zip(rows,CONTENT['topics']):
        result=hyso_guide.show_topic(row['id'],lang)
        assert result['id']==row['id']==source['id']
        assert result['title']==row['title']==source['translations'][lang]['title']
        for value in source['translations'][lang].values():
            assert value in result['text']
        for other in ('tr','en','de','es'):
            if other!=lang and source['translations'][other]['title']!=result['title']:
                assert source['translations'][other]['title'] not in result['text']
        uri=urlsplit(result['uri'])
        assert uri.scheme=='file' and uri.fragment==row['id'] and uri.query=='lang='+lang
        assert unquote(uri.path).endswith('/hyso/data/user-guide.html')
        assert result['related']==source['related']
        if 'code' in source:assert source['code'] in result['text']


@pytest.mark.parametrize('symbol,topic',[
    ('hyso.ModelBuilder','models.model-builder'),
    ('hyso.GraphModelBuilder','models.graph-builder'),
    ('hyso.TrainingConfig','training.configuration'),
    ('hyso.TrainingSession','training.configuration'),
    ('hyso.Trainer','training.trainer'),
    ('hyso.analysis.plan_analysis','analyzer.workflow'),
    ('hyso.analysis.compare_models','comparison.models'),
    ('hyso.advisor.recommend','advisor.recommendations'),
    ('hyso.reporting.create_report','reporting.outputs'),
    ('hyso.ExperimentManager','experiments.manager'),
    ('hyso.performance.measure_training_memory_isolated','analysis.p23'),
    ('hyso.performance.measure_throughput','analysis.p02'),
    ('registry:conv2d','components.convolution'),
    ('registry:modular_cnn','models.support'),
])
def test_api_links_match_the_actual_model_or_analysis(symbol,topic):
    assert hyso_guide.topic_for_api(symbol)==topic
    assert hyso_guide.guide_uri(topic,'de').endswith('?lang=de#'+topic)


def test_index_version_all_topics_and_source_docstrings():
    assert INDEX['hyso_version']==TOPICS['hyso_version']=='0.2.0'
    valid={t['id'] for t in TOPICS['topics']}
    trees={};seen=set()
    for symbol,entry in INDEX['entries'].items():
        assert entry['topic'] in valid,(symbol,entry)
        if 'path' not in entry:continue  # Upstream PyTorch docs are preserved.
        key=(entry['path'],entry['qualname'])
        if key in seen:continue
        seen.add(key)
        if entry['path'] not in trees:
            trees[entry['path']]=ast.parse((ROOT/entry['path']).read_text(encoding='utf-8'),feature_version=(3,10))
        body=trees[entry['path']].body
        def declarations(nodes):
            for declaration in nodes:
                if isinstance(declaration,ast.FunctionDef) and any(
                    isinstance(d,ast.Attribute) and d.attr=='overload' for d in declaration.decorator_list):
                    continue
                if isinstance(declaration,(ast.ClassDef,ast.FunctionDef)):yield declaration
                elif isinstance(declaration,ast.If):
                    yield from declarations(declaration.body)
                    yield from declarations(declaration.orelse)
        for part in entry['qualname'].split('.'):
            node=next(n for n in declarations(body) if n.name==part)
            body=node.body
        doc=ast.get_docstring(node) or ''
        assert 'HYSo Guide: '+entry['topic'] in doc,key
        assert 'hyso docs show '+entry['topic'] in doc,key
        for field in ('Returns:','Raises:','Example'):
            assert field in doc,(key,field)
        assert not doc.startswith('Use the existing'),key
        assert not doc.startswith('Configure '+node.name+'.'),key
        assert 'input; native validation and scope remain unchanged.' not in doc,key
        if isinstance(node,ast.FunctionDef):
            args=[*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs]
            args += [a for a in (node.args.vararg,node.args.kwarg) if a]
            assert all(a.annotation is not None for a in args if a.arg not in ('self','cls')),key
            assert node.returns is not None,key
    assert len(seen)>=597


def test_public_facades_and_supported_component_inventory_are_covered():
    for namespace in ('hyso.analysis','hyso.advisor','hyso.reporting','hyso.benchmark'):
        module=importlib.import_module(namespace)
        for name in module.__all__:
            if inspect.isclass(getattr(module,name)) or inspect.isfunction(getattr(module,name)):
                assert namespace+'.'+name in INDEX['entries'],(namespace,name)
    from hyso.models.registry import ModelRegistry
    for row in ModelRegistry().list():assert 'registry:'+row.name in INDEX['entries']
    for name in ('add','concat','multiply','weighted_sum'):
        assert hyso_guide.topic_for_api('graph:'+name)=='components.fusion'


def test_hints_resolve_without_constructing_models():
    import torch
    from torch.utils.data import Dataset,DataLoader
    from hyso.training import TrainingSession
    seen=set()
    for entry in INDEX['entries'].values():
        if 'path' not in entry:continue
        key=(entry['path'],entry['qualname'])
        if key in seen:continue
        seen.add(key)
        module=importlib.import_module(entry['path'][:-3].replace('/','.'))
        obj=module
        for part in entry['qualname'].split('.'):obj=getattr(obj,part)
        if isinstance(obj,property):obj=obj.fget
        # TrainingConfig intentionally keeps these imports under TYPE_CHECKING.
        if entry['qualname'] in ('TrainingConfig.make_loaders','TrainingConfig.prepare','TrainingConfig.prepare_datasets'):
            namespace=dict(vars(module),nn=torch.nn,Dataset=Dataset,DataLoader=DataLoader,TrainingSession=TrainingSession)
            typing.get_type_hints(obj,globalns=namespace)
        else:
            typing.get_type_hints(obj)


def test_record_collection_hints_and_export_returns_are_accurate(tmp_path):
    from hyso.analysis import EvidenceIndex, EvidenceRecord, UnifiedAnalysisReport, AnalysisContext
    from hyso.advisor import NativeMetricResolver
    from hyso.performance.roofline_counters import analyze_counter_roofline
    assert typing.get_args(typing.get_type_hints(EvidenceIndex.__init__)['records'])==(EvidenceRecord,)
    assert typing.get_args(typing.get_type_hints(UnifiedAnalysisReport.build)['risks'])==(str,)
    assert typing.get_type_hints(UnifiedAnalysisReport.build)['return'] is UnifiedAnalysisReport
    assert typing.get_type_hints(UnifiedAnalysisReport.to_markdown)['return'] is type(None)
    assert typing.get_args(typing.get_type_hints(NativeMetricResolver.check_reference)['required_native'])==(str,)
    assert typing.get_args(typing.get_type_hints(AnalysisContext.__init__)['references'])==(typing.Any,)
    assert typing.get_args(typing.get_type_hints(analyze_counter_roofline)['compute_sessions'])==(dict[str,typing.Any],)
    from hyso.experiments.manager import ExperimentManager,ExperimentRun
    assert typing.get_type_hints(ExperimentManager.create_run)['return'] is ExperimentRun
    assert typing.get_type_hints(ExperimentManager.get_run)['return'] is ExperimentRun
    assert typing.get_type_hints(ExperimentManager.get_experiment)['id'] is str
    assert typing.get_type_hints(ExperimentRun.id.fget)['return'] is str
    assert typing.get_type_hints(ExperimentRun.log_metric)['return']==dict[str,typing.Any]
    manager=ExperimentManager(tmp_path/'store')
    try:
        experiment=manager.create_experiment('hover-type-check')
        run=manager.create_run(experiment['id'],'training',{})
        assert type(run) is ExperimentRun and type(manager.get_run(run.id)) is ExperimentRun
        run.start()
        record=run.log_metric(name='observed_loss',value=0.5,unit='loss',source='explicit_observation',epoch=1)
        assert type(record) is dict and record['value']==0.5
        run.finish()
    finally:manager.close()


def test_generated_class_examples_bind_to_real_signatures():
    seen=set()
    for entry in INDEX['entries'].values():
        if entry.get('kind')!='class' or 'path' not in entry or entry['canonical'] in seen:continue
        seen.add(entry['canonical'])
        module=importlib.import_module(entry['path'][:-3].replace('/','.'))
        obj=module
        for part in entry['qualname'].split('.'):obj=getattr(obj,part)
        doc=inspect.getdoc(obj) or ''
        marker='# Caller-supplied values; explicitly requested operations only.'
        if marker not in doc:continue
        import textwrap
        line=textwrap.dedent(doc.split(marker,1)[1].split('\n\n',1)[0]).strip()
        sample=ast.parse(line).body[0]
        call=sample.value if isinstance(sample,(ast.Expr,ast.Assign)) else None
        if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id==obj.__name__:
            # Bind placeholders; do not evaluate expressions or construct the class.
            try:signature=inspect.signature(obj)
            except ValueError:
                assert issubclass(obj,BaseException)  # Builtin exception constructors expose no inspect signature.
                continue
            signature.bind(*[object() for a in call.args],**{k.arg:object() for k in call.keywords})


def test_missing_unknown_and_invalid_language_fail_without_browser(capsys,monkeypatch):
    from hyso.cli import main
    monkeypatch.setattr(webbrowser,'open',lambda *a:pytest.fail('invalid input opened a browser'))
    for args in (['docs','show'],['docs','open','bad.topic'],['docs','show','analysis.p23','--lang','xx']):
        assert main(args)==2
        captured=capsys.readouterr()
        assert captured.err and 'Traceback' not in captured.err
    for value in ('hyso.Unknown','hyso.Analyzer','not.an.api'):
        with pytest.raises(ValueError):hyso_guide.topic_for_api(value)
    with pytest.raises(ValueError):hyso_guide.guide_uri('analysis.p99')
    with pytest.raises(ValueError):hyso_guide.list_topics('xx')


def test_cli_read_only_and_explicit_browser_contract(capsys,monkeypatch):
    from hyso.cli import main
    opened=[];monkeypatch.setattr(webbrowser,'open',lambda uri:opened.append(uri) or True)
    for args in (['docs','list'],['docs','show','models.graph-builder','--lang','tr'],
                 ['docs','open','analysis.p23','--lang','de','--dry-run']):
        assert main([*args,'--json'])==0
        value=json.loads(capsys.readouterr().out)
        assert value['format']=='hyso.cli' and value['schema_version']==1
        assert not opened
    assert value['status']=='dry_run' and value['result']['opened'] is False
    assert main(['docs','open','analysis.p23','--lang','de'])==0
    assert opened==[hyso_guide.guide_uri('analysis.p23','de')]
    assert opened[0] in capsys.readouterr().out
    monkeypatch.setattr(webbrowser,'open',lambda uri:False)
    assert main(['docs','open','models.model-builder'])==2
    assert 'manually' in capsys.readouterr().err
    def failed(uri):raise webbrowser.Error('no browser available')
    monkeypatch.setattr(webbrowser,'open',failed)
    assert main(['docs','open','models.graph-builder'])==2
    assert 'no browser available' in capsys.readouterr().err


def test_cli_debug_preserves_resource_error(capsys,monkeypatch):
    from hyso.cli import main
    def broken(*args):raise FileNotFoundError('missing installed guide resource')
    monkeypatch.setattr(hyso_guide,'_resources',broken)
    assert main(['docs','show','analysis.p23'])==2
    err=capsys.readouterr().err
    assert 'missing installed guide resource' in err and 'Traceback' not in err
    assert main(['docs','show','analysis.p23','--debug'])==2
    assert 'Traceback' in capsys.readouterr().err


def test_clean_wheel_topic_reader_and_cli(distributions,tmp_path):
    work,wheel,_=distributions
    target=tmp_path/'installed';venv.EnvBuilder(with_pip=False,system_site_packages=True).create(target)
    exe=target/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    env=dict(os.environ,PIP_CONFIG_FILE=os.devnull,PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8')
    env.pop('PYTHONPATH',None)
    def run(argv):
        result=subprocess.run(argv,cwd=work,env=env,capture_output=True,text=True,encoding='utf-8',timeout=90)
        assert result.returncode==0,result.stdout+result.stderr
        return result.stdout
    run([sys.executable,'-m','pip','--python',str(exe),'install','--no-index','--no-deps',str(wheel)])
    script='''import pathlib,sys,webbrowser
import hyso_guide
assert pathlib.Path(hyso_guide.__file__).is_relative_to(pathlib.Path(sys.prefix))
assert "hyso" not in sys.modules and "torch" not in sys.modules
assert hyso_guide.topic_for_api("hyso.GraphModelBuilder")=="models.graph-builder"
assert hyso_guide.topic_for_api("hyso.performance.measure_training_memory_isolated")=="analysis.p23"
assert "?lang=de#analysis.p23" in hyso_guide.guide_uri("analysis.p23","de")
topic=hyso_guide.show_topic("models.graph-builder","es")
assert topic["lang"]=="es" and "\\n    .add(" in topic["text"]
import torch
def forbidden(*a,**k): raise AssertionError("installed help started a workload")
torch.nn.Module.__init__=forbidden
torch.cuda._lazy_init=forbidden
torch.cuda.is_available=forbidden
webbrowser.open=forbidden
from hyso.cli import main
from hyso.experiments.manager import ExperimentManager,ExperimentRun
from typing import get_type_hints
assert get_type_hints(ExperimentManager.create_run)["return"] is ExperimentRun
assert get_type_hints(ExperimentRun.log_metric)["return"]==dict[str,__import__("typing").Any]
assert main(["docs","show","analysis.p23","--lang","de","--json"])==0
assert main(["docs","open","models.graph-builder","--lang","tr","--dry-run"])==0
assert main(["models","list","--json"])==0
'''
    output=run([str(exe),'-I','-B','-c',script])
    assert 'analysis.p23' in output and 'models.graph-builder' in output
    console=target/('Scripts/hyso.exe' if os.name=='nt' else 'bin/hyso')
    assert 'analysis.p23' in run([str(console),'docs','list','--lang','de'])
