"""Render retained Advisor reports only; never recompute analysis or planning."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--legacy-markdown',type=Path,required=True)
    parser.add_argument('--goal-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    import torch
    assert not torch.cuda.is_initialized()
    calls=[]
    def deny(label):
        def blocked(*a,**k):calls.append(label);raise AssertionError('Forbidden reporting call:'+label)
        return blocked
    for name in ('init','is_available','device_count','current_device'):
        setattr(torch.cuda,name,deny(name))
    torch.nn.Module._call_impl=deny('model.forward')
    def audit(event,values):
        if event=='import' and values[0] in ('cnn_factories','hyso.models.modular'):
            calls.append(values[0]);raise AssertionError('Source model/factory import forbidden')
    sys.addaudithook(audit)
    from hyso.benchmark import BenchmarkRunner
    from hyso.training import Trainer
    BenchmarkRunner.run=deny('BenchmarkRunner.run');Trainer.fit=deny('Trainer.fit')
    from hyso.advisor import AdvisorError,AdvisorReport
    from hyso.advisor.presentation import build_display_model
    import hyso.advisor.engine as engine
    import hyso.advisor.goals as goals
    from hyso.advisor.resolver import NativeMetricResolver
    engine.advise=deny('advise');engine._advise_validated=deny('_advise_validated')
    goals.evaluate_goal=deny('evaluate_goal');goals.evaluate_goals=deny('evaluate_goals')
    NativeMetricResolver.__init__=deny('NativeMetricResolver')
    inputs=(args.report,args.legacy_markdown,args.goal_report)
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    report=AdvisorReport.from_json(args.report);goal_report=AdvisorReport.from_json(args.goal_report)
    canonical_before=report.to_json();report_id=report.report_id
    checks=[]
    def check(name,condition):
        assert condition,name
        checks.append(dict(check=name,status='PASS'))
    check('Canonical JSON exact previous bytes',canonical_before.encode('utf-8')==args.report.read_bytes())
    check('Default Markdown exact historical bytes',report.to_markdown().encode('utf-8')==args.legacy_markdown.read_bytes())
    model=build_display_model(report)
    args.output.mkdir(parents=True,exist_ok=False)
    for language in ('tr','en'):
        for audience in ('summary','technical'):
            stem='mnist-'+language+'-'+audience
            markdown=args.output/(stem+'.md');display=args.output/(stem+'.display.json')
            report.to_markdown(markdown,language=language,audience=audience)
            report.to_display_json(display,language=language,audience=audience)
            view=json.loads(display.read_text(encoding='utf-8'))
            check(stem+' shares exact language-neutral semantic data',view['data']==model)
            check(stem+' is presentation-only and references original identity',view['presentation_only']
                  and not view['authoritative_evidence'] and view['source_advisor_report_id']==report_id)
            check(stem+' UTF8 Markdown/export deterministic',markdown.read_text(encoding='utf-8')==
                  report.to_markdown(language=language,audience=audience))
            if audience=='technical':
                check(stem+' contains complete original technical record',
                      view['data']['original_technical_record']==json.loads(canonical_before)
                      and 'metric_path' in markdown.read_text(encoding='utf-8'))
    tr=report.to_display_dict(language='tr');en=report.to_display_dict(language='en')
    check('Both languages preserve all metric values/references/statuses',tr['data']['metrics']==en['data']['metrics'])
    localized={m['metric']:m for m in tr['text']['metrics']}
    check('Actual P95/E2E/throughput/accuracy rendered from existing metrics',
          localized['resident_forward_p95']['value']=='0,86681 ms' and localized['e2e_p95']['value']=='4,642905 ms'
          and localized['completed_images_per_second']['value']=='12.442,17 görüntü/sn'
          and localized['test_accuracy']['value']=='%98,9')
    check('Native 18 observations and 7 next-evidence findings preserved',model['counts']['OBSERVATION']==18 and model['counts']['MEASURE_NEXT']==7)
    check('No fabricated candidates/verified improvement',model['counts']['EXPERIMENT_CANDIDATE']==0 and model['counts']['VERIFIED_IMPROVEMENT']==0)
    goal_tr=goal_report.to_display_dict(language='tr');goal_en=goal_report.to_display_dict(language='en')
    check('Goal outcomes and hard constraints identical across languages',goal_tr['data']['goals']==goal_en['data']['goals']
          and goal_tr['data']['original_technical_record']==goal_report.to_dict())
    check('P18/P29/P31/B4 original boundaries and Turkish meanings retained',tr['data']['original_risks']==list(report.risks)
          and all(any(term in risk for risk in tr['text']['risks']) for term in ('P18','P29','P31 7200','B4 native çökme riski açık')))
    check('Turkish explanations are localized, not headings only',
          'Mevcut kanıtlar henüz bir optimizasyon deneyi önermek için yeterli değil.' in report.to_markdown(language='tr')
          and 'No optimization experiment is sufficiently supported yet.' not in report.to_markdown(language='tr'))
    check('Unsupported language explicitly rejected',rejected_language(report))
    check('Canonical identity and bytes unchanged after all views',report.to_json()==canonical_before and report.report_id==report_id)
    check('Display JSON rejected as canonical evidence',rejected_display(tr))
    check('Existing source/evidence bytes unchanged',all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items()))
    check('No engine/model/factory/device/training/benchmark calls',not calls and not torch.cuda.is_initialized())
    result=dict(status='PASS',passed=len(checks),failed=0,checks=checks,source_advisor_report_id=report_id,
                canonical_file_sha256=before[str(args.report)],input_sha256=before,
                finding_counts=model['counts'],forbidden_calls=calls,cuda_initialized=False,
                output_files=sorted(p.name for p in args.output.iterdir()))
    (args.output/'acceptance.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))


def rejected_language(report):
    from hyso.advisor import AdvisorError
    try:report.to_display_json(language='fr')
    except AdvisorError:return True
    return False


def rejected_display(view):
    from hyso.advisor import AdvisorError,AdvisorReport
    try:AdvisorReport.from_dict(view)
    except AdvisorError:return True
    return False


if __name__=='__main__':main()
