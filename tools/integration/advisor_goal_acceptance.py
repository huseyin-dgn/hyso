"""Retained MNIST goal acceptance and exact 3B.2 Automatic regression check."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--baseline-advisor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    assert not torch.cuda.is_initialized()
    calls = []
    def deny(label):
        def blocked(*a, **k):
            calls.append(label)
            raise AssertionError('Forbidden runtime call:' + label)
        return blocked
    for name in ('init', 'is_available', 'device_count', 'current_device'):
        setattr(torch.cuda, name, deny(name))
    torch.nn.Module._call_impl = deny('model.forward')
    def audit(event, values):
        if event == 'import' and values[0] in ('cnn_factories', 'hyso.models.modular'):
            calls.append(values[0]); raise AssertionError('Source model/factory import forbidden')
    sys.addaudithook(audit)
    from hyso.benchmark import BenchmarkRunner
    from hyso.training import Trainer
    BenchmarkRunner.run = deny('BenchmarkRunner.run'); Trainer.fit = deny('Trainer.fit')
    from hyso.analysis import UnifiedAnalysisReport
    from hyso.advisor import (AdvisorReport, AdvisorRequest, Metric, NativeMetricResolver,
        UserGoal, advise, evaluate_goal, goal_context, validate_advisor_report, validate_request)
    from hyso.advisor.engine import _advise_validated
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.source, args.baseline_advisor)}
    source = UnifiedAnalysisReport.from_json(args.source)
    baseline = AdvisorReport.from_json(args.baseline_advisor)
    resolver = NativeMetricResolver(source)
    automatic = advise(source)
    checks = []
    def check(name, condition):
        assert condition, name
        checks.append(dict(check=name, status='PASS'))
    check('Automatic exact 3B.2 JSON identity', automatic.to_dict() == baseline.to_dict())
    check('Automatic exact Markdown', automatic.to_markdown() == baseline.to_markdown())
    check('Automatic 18 observations / 2 next-evidence findings',
          sum(f.kind == 'OBSERVATION' for f in automatic.findings) == 18 and
          sum(f.kind == 'MEASURE_NEXT' for f in automatic.findings) == 2)
    record = next(r for r in resolver.report['index']['records'] if r['feature_id'] == 'P01')
    scope, protocol = goal_context(resolver, record['record_id'], Metric.RESIDENT_FORWARD_P95)
    normal = UserGoal('normal-p95', Metric.RESIDENT_FORWARD_P95, '<=', 1, 'ms', scope=scope, protocol=protocol)
    tough = replace(normal, goal_id='aggressive-p95', target=.001)
    lower = replace(normal, goal_id='lower-p95', operator='>=', target=2)
    normal_result, tough_result = evaluate_goal(resolver, normal), evaluate_goal(resolver, tough)
    check('Normal target locally met', normal_result.observed == 'observed_met' and normal_result.input_validity == 'valid')
    check('Aggressive positive target valid but locally missed', tough_result.observed == 'observed_missed' and tough_result.input_validity == 'valid')
    check('No future or deployment certification', normal_result.feasibility == tough_result.feasibility == 'unknown'
          and all(any('deployment_acceptance:not_evaluated' in r for r in e.reasons) for e in (normal_result, tough_result)))
    check('Bindings preserved', {b.field:b.authority for b in normal_result.evidence[0].bindings} == record['bindings'])
    request = AdvisorRequest([normal, tough, lower])
    check('Same-scope hard contradiction', validate_request(request).consistency == 'logically_inconsistent')
    other_batch = replace(lower, scope=replace(scope, batch=2, input_shape=(2, 1, 28, 28)))
    check('Different batch not a mathematical contradiction', validate_request(AdvisorRequest([normal, other_batch])).consistency == 'no_conflict_identified')
    check('Different batch cannot claim target met', evaluate_goal(resolver, other_batch).observed == 'unknown')
    missing = replace(normal, scope=None)
    check('Missing scope not inferred', evaluate_goal(resolver, missing).input_validity == 'needs_scope')
    advanced = advise(source, request=request)
    check('Public advise uses real goal engine', advanced.evaluation_status == 'goals_evaluated'
          and tuple(g.observed.value for g in advanced.goal_evaluations) == ('observed_met', 'observed_missed', 'observed_missed'))
    for priority in ('speed', 'memory', 'preserve_accuracy', 'balanced'):
        # Reuse the already validated immutable snapshot; public entry point and
        # all priority routes also have focused fixture coverage.
        variant = _advise_validated(resolver, replace(request, priority=priority))
        check('Priority ' + priority + ' leaves goals and evidence unchanged',
              variant.goal_evaluations == advanced.goal_evaluations and variant.request.goals == advanced.request.goals
              and {f.finding_id:f.to_dict() for f in variant.findings} == {f.finding_id:f.to_dict() for f in advanced.findings})
    check('No prohibited finding types', all(f.kind in ('OBSERVATION', 'MEASURE_NEXT') for f in advanced.findings))
    risks = '\n'.join(advanced.risks)
    check('P18/P29/P31/B4 boundaries unchanged', all(s in risks for s in (
        'qualified_stable_roof=not_established_by_this_report', 'independently_verified_improvement=not_certified',
        'long_run_7200_acceptance=not_certified', 'B4 native risk remains OPEN')))
    args.output.mkdir(parents=True, exist_ok=False)
    advanced.to_json(args.output / 'mnist-goals.json')
    advanced.to_markdown(args.output / 'mnist-goals.md')
    reopened = AdvisorReport.from_json(args.output / 'mnist-goals.json')
    check('Goal JSON/Markdown roundtrip', reopened == advanced and reopened.to_markdown() == advanced.to_markdown())
    check('Goal derivation against source', validate_advisor_report(reopened, source))
    check('Existing source/baseline bytes preserved', all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p,sha in before.items()))
    check('No model/training/benchmark/factory/device calls', not calls and not torch.cuda.is_initialized())
    summary = dict(status='PASS', passed=len(checks), failed=0, checks=checks,
        automatic_report_id=automatic.report_id, goal_report_id=advanced.report_id,
        goal_results=[g.to_dict() for g in advanced.goal_evaluations],
        consistency=advanced.request_validation.consistency.value,
        preserved_input_sha256=before, forbidden_calls=calls, cuda_initialized=False)
    (args.output/'acceptance.json').write_text(json.dumps(summary, indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k != 'goal_results'},indent=2))


if __name__ == '__main__':
    main()
