"""Public API and retained-evidence checks; never creates or executes a model."""
import contextlib
import importlib
import inspect
import io
import json
import pydoc
import typing
from pathlib import Path
from unittest.mock import patch

from tools.product_acceptance.inventory import REPO, USER, build_inventory, read


def require(condition, message):
    if not condition:
        raise AssertionError(message)


@contextlib.contextmanager
def offline_only():
    import torch
    # Initialize Torch's function-identity registry before patching its query
    # functions. Importing torchvision does not construct or execute a model.
    import torchvision
    def forbidden(*args, **kwargs):
        raise RuntimeError('Product experience checks must not start workloads or query CUDA')
    with contextlib.ExitStack() as stack:
        for target, names in ((torch.nn.Module, ('__init__', '_call_impl')), (torch, ('load',)),
                (torch.cuda, ('init', '_lazy_init', 'is_available', 'device_count', 'synchronize', 'current_device'))):
            for name in names:
                stack.enter_context(patch.object(target, name, forbidden))
        yield


def U15(folder):
    import hyso
    require(len(hyso.__all__)==len(set(hyso.__all__)), 'duplicate root exports')
    require(all(not n.startswith('_') and n in dir(hyso) for n in hyso.__all__), 'private/undiscoverable export')
    for name in hyso.__all__:
        getattr(hyso, name)
    require(hyso.models.__all__==['EBO','ELO','CEMO','EMA','CNNSENormalWithFit'], 'legacy exports changed')
    modules=('hyso.evaluation','hyso.quality','hyso.advisor.validation','hyso.advisor.goals',
        'hyso.advisor.comparison','hyso.advisor.performance','hyso.advisor.profiles','hyso.advisor.presentation')
    returns={}
    for module in modules:
        m=importlib.import_module(module)
        for name, value in vars(m).items():
            if inspect.isfunction(value) and not name.startswith('_') and value.__module__==module:
                hints=typing.get_type_hints(value)
                require('return' in hints, module+'.'+name+' missing return type')
                returns[module+'.'+name]=str(hints['return'])
    for name in ('reopen_stability','measure_streaming','compare_devices','profile_execution_providers'):
        value=getattr(hyso.performance,name)
        require(typing.get_type_hints(value)['return'].__name__=='RuntimeResult', name+' return contract')
    require((REPO/'hyso/py.typed').is_file(), 'missing py.typed')
    require('py.typed' in (REPO/'setup.cfg').read_text(), 'package typing metadata')
    return dict(exports=hyso.__all__, concrete_returns=returns, real_IDE_tested=False,
        type_checker='not installed/not run', limitations='Dynamic metadata/factories retain Any; no strict package typing claim')


def U16(folder):
    import hyso
    import hyso_guide
    from hyso.cli import main
    helps={}
    for name in ('ModelBuilder','GraphModelBuilder','Trainer','evaluate_classification','evaluate_precision_quality'):
        value=getattr(hyso,name)
        text=pydoc.render_doc(value)
        require(len(text)>100 and inspect.getdoc(value), name+' help absent')
        helps[name]=str(inspect.signature(value))
    commands=([],['models'],['train'],['analysis'],['advisor'],['reports'],['compare'],['docs'])
    for command in commands:
        with contextlib.redirect_stdout(io.StringIO()) as capture:
            try:
                code=main([*command,'--help'])
            except SystemExit as exc:
                code=exc.code
        require(code==0 and 'usage:' in capture.getvalue(), 'CLI help '+str(command))
    counts={}
    for lang in ('tr','en','de','es'):
        rows=hyso_guide.list_topics(lang);counts[lang]=len(rows)
        for row in rows:
            require(hyso_guide.show_topic(row['id'],lang)['text'], 'empty topic')
    require(set(counts.values())=={70}, 'guide coverage')
    try:
        hyso_guide.show_topic('analysis.p01','fr')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown language accepted')
    return dict(signatures=helps, guide_topics=counts, Python_help='primarily English', CLI_help='primarily English',
        localized_surfaces='guide, report, Advisor and comparison presentation: tr/en/de/es')


def retained(scenario, name):
    cp=read(USER/'checkpoint.json')
    record=cp['cpu_results']['tests/test_user_acceptance_scenarios.py::test_'+scenario]
    require(record['status']=='PASS' and record.get('source_valid',True), 'invalid retained source')
    return read(Path(record['evidence']).parent/scenario/name)


def presentation_projection(body):
    return [dict(kind=s['kind'],raw=s['raw'], source_id=s['source_id'],
        metrics=[{k:m.get(k) for k in ('key','value','unit','status','scope','feature_id','record_id')} for m in s['metrics']])
        for s in body['sources']]


def validate_manual_advice(body, source):
    require(body['claims']==dict(global_winner=None,numerical_improvement=None,automatically_applied=False), 'unmeasured claims')
    ids={r['record_id'] for r in source['index']['records']}
    items=body['recommendations']
    require(bool(items) and any(i['retained_record_ids'] for i in items), 'no evidence-linked advice')
    require(all(set(i['retained_record_ids'])<=ids and i['estimated_improvement'] is None and
        i['manual_only'] is True and i['verified_effect']=='not_measured' for i in items), 'unbound or fabricated recommendation')


def U17(folder):
    from hyso import analysis, advisor, reporting, QualityExperimentResult
    raw=retained('U09','advisor-raw.json')
    source=analysis.UnifiedAnalysisReport(raw['source'])
    before=source.to_dict()
    advice=advisor.recommend(source)
    validate_manual_advice(advice.to_dict(), before)
    comparison=retained('U08','two-models-raw.json')
    quality=QualityExperimentResult(comparison['comparison'])
    require(comparison['source_weights_before']==comparison['source_weights_after'] and comparison['same_cohort'], 'comparison provenance')
    baseline=None;explanations={};outputs=[]
    for lang in ('tr','en','de','es'):
        rows=reporting.feature_explanations(language=lang)
        require(len(rows)==35 and len({r['feature_id'] for r in rows})==35, '35 explanations')
        for r in rows:
            require(all(r[k] for k in ('definition','purpose','units','limits','explanations','guide_url')), 'incomplete explanation '+r['feature_id'])
        explanations[lang]=[{k:r[k] for k in ('feature_id','units')} for r in rows]
        report=reporting.create_report(source,advice,language=lang)
        body=report.to_dict();projection=presentation_projection(body)
        if baseline is None:baseline=projection
        require(projection==baseline, 'localized numbers/units/evidence changed')
        path=folder/('report-'+lang+'.json');report.to_json(path)
        require(reporting.PresentationReport.from_json(path).to_dict()==body, 'report roundtrip')
        report.to_markdown(folder/('report-'+lang+'.md'))
        report.to_html(folder/('report-'+lang+'.html'))
        outputs.append(str(path))
    require(all(v==explanations['en'] for v in explanations.values()), 'localized units/IDs changed')
    require(source.to_dict()==before and quality.to_dict()==comparison['comparison'], 'retained data mutated')
    try:
        reporting.create_report(source,language='fr')
    except ValueError:
        pass
    else:raise AssertionError('unknown report language accepted')
    return dict(retained_source=before, advice=advice.to_dict(), explanations=explanations,outputs=outputs,
        comparison='validated retained P29; no workload rerun', limits='Resolver subset only; no global winner or guaranteed gains')


def U18(folder):
    actual=build_inventory();expected=read(REPO/'docs/acceptance/product-experience-inventory.json')
    require(actual==expected, 'inventory differs from prepared source/evidence')
    require(len(actual['components'])==21 and len(actual['performance_features'])==35 and len(actual['exports'])==10, 'inventory coverage')
    require(not actual['full_cuda_acceptance'] and len(actual['deferred'])==4, 'deferred scope changed')
    for row in actual['components']:
        require(row['implemented'] and row['guide']['present'], 'missing source/guide '+row['feature'])
    for row in actual['exports']:
        require(row['hyso_conversion_api']=='MISSING', 'unimplemented converter claimed')
    return actual


def execute(folder, name):
    folder=Path(folder)/name;folder.mkdir(parents=True,exist_ok=True)
    result=dict(scenario=name,public_api='existing public discovery/help/retained presentation',device='cpu',
        expected='offline source/evidence contracts',status='FAIL',actual=None)
    try:
        with offline_only():
            raw=globals()[name](folder)
        (folder/'raw.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
        result.update(status='PASS',actual='contracts verified',raw_evidence_path=str(folder/'raw.json'))
    except Exception as exc:
        result['actual']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

