"""Read-only comparison, recommendations and report presentation commands."""
from pathlib import Path
from .common import (Outcome, command, output_check, export_text, json_text, read_result,
                     presentation_options, analysis_context, existing_manager, require)


def register(sub):
    q = command(sub, 'compare', 'Compare two retained analysis/execution files using ModelComparison rules; never run P29 or declare a global winner.', run)
    q.add_argument('baseline', type=Path)
    q.add_argument('candidate', type=Path)
    q.add_argument('--baseline-model', required=True, help='Explicit baseline model label in retained evidence.')
    q.add_argument('--candidate-model', required=True, help='Explicit candidate model label in retained evidence.')
    q.add_argument('--features', nargs='+', required=True, help='Explicit comparison feature selection, e.g. P01 P02 P11 P26.')
    presentation_options(q, formats=('json', 'markdown'), levels=False)
    p = command(sub, 'advisor', 'Generate manual recommendations from validated retained evidence; no capture or optimization.', run, export=False)
    s = p.add_subparsers(dest='action', required=True)
    q = command(s, 'recommend', 'Call hyso.advisor.recommend on a saved UnifiedAnalysisReport, EvidenceIndex or ModelComparisonResult.', run)
    q.add_argument('input', type=Path)
    q.add_argument('--context', type=Path, help='AnalysisContext keyword JSON required for EvidenceIndex inputs; caller declarations only.')
    presentation_options(q)
    p = command(sub, 'reports', 'Present validated saved results using hyso.reporting; no measurements.', run, export=False)
    s = p.add_subparsers(dest='action', required=True)
    q = command(s, 'render', 'Render TrainingResult, UnifiedAnalysisReport, execution, EvidenceIndex, comparison or Advisor recommendations.', run)
    q.add_argument('input', type=Path)
    q.add_argument('--context', type=Path, help='AnalysisContext keyword JSON required for execution/EvidenceIndex inputs.')
    q.add_argument('--advisor-source', type=Path, help='Original UnifiedAnalysisReport required to validate a raw AdvisorReport; not needed for recommendations.')
    presentation_options(q)
    q = command(s, 'experiment', 'Use report_experiment over all retained experiment history pages; no migration or artifact repair.', run)
    q.add_argument('--root', required=True, type=Path)
    q.add_argument('id', help='Experiment ID.')
    presentation_options(q)


def _comparison(args):
    from hyso.analysis import ModelComparisonSource, AnalysisExecutionResult, UnifiedAnalysisReport, compare_models
    def source(path, model):
        result = read_result(path)
        if type(result) is AnalysisExecutionResult:
            return ModelComparisonSource.from_execution(result, model)
        require(type(result) is UnifiedAnalysisReport, 'compare accepts validated execution or UnifiedAnalysisReport JSON files.')
        return ModelComparisonSource(model, result)
    return compare_models(source(args.baseline, args.baseline_model), source(args.candidate, args.candidate_model), features=args.features)


def run(args):
    from hyso.reporting import create_report, report_experiment
    output_check(args, inputs=[getattr(args, n, None) for n in ('input', 'context', 'baseline', 'candidate', 'advisor_source')])
    if args.group == 'compare':
        result = _comparison(args)
        raw = result.to_json() if args.format == 'json' else result.to_markdown(language=args.language)
        data = result.to_dict()
    elif args.group == 'advisor':
        from hyso.advisor import recommend
        result = recommend(read_result(args.input), context=analysis_context(args.context))
        raw = result.to_json() if args.format == 'json' else getattr(result, 'to_' + args.format)(language=args.language, level=args.level)
        data = result.to_display_dict(language=args.language, level=args.level) if args.format == 'json' else result.to_dict()
    else:
        if args.action == 'experiment':
            if args.output is not None:
                require(not args.output.resolve().is_relative_to(args.root.resolve()), 'Export outside the ExperimentManager store directory.')
            manager = existing_manager(args.root)
            try:
                result = report_experiment(manager, args.id, language=args.language, level=args.level)
            finally:
                manager.close()
        else:
            source = read_result(args.input)
            from hyso.advisor import AdvisorReport, validate_advisor_report
            if type(source) is AdvisorReport:
                require(args.advisor_source is not None, 'Raw AdvisorReport rendering requires --advisor-source with its original UnifiedAnalysisReport.')
                validate_advisor_report(source, read_result(args.advisor_source))
            else:
                require(args.advisor_source is None, '--advisor-source applies only to raw AdvisorReport inputs.')
            result = create_report(source, context=analysis_context(args.context),
                                   language=args.language, level=args.level)
        raw = getattr(result, 'to_' + args.format)()
        data = result.to_dict()
    export_text(args, raw)
    return Outcome(data, text=raw)
