"""Read retained MNIST evidence, compare prior reports, export proposed plans only."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--observations-baseline', type=Path, required=True)
    parser.add_argument('--goals-baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    assert not torch.cuda.is_initialized()
    calls = []
    def deny(label):
        def blocked(*a, **k):
            calls.append(label); raise AssertionError('Forbidden runtime call:' + label)
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
    from hyso.advisor import (AdvisorReport, AdvisorRequest, NativeMetric as M, NativeMetricResolver,
                             NativeObservation, PlannedFinding, advise, validate_advisor_report)
    from hyso.advisor.engine import _advise_validated
    paths = (args.source, args.observations_baseline, args.goals_baseline)
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    source = UnifiedAnalysisReport.from_json(args.source)
    old = AdvisorReport.from_json(args.observations_baseline)
    old_goals = AdvisorReport.from_json(args.goals_baseline)
    resolver = NativeMetricResolver(source)
    automatic = advise(source)
    checks = []
    def check(name, condition):
        assert condition, name
        checks.append(dict(check=name, status='PASS'))
    def counts(report):
        found = Counter(f.kind.value for f in report.findings)
        return {k:found[k] for k in ('OBSERVATION','HYPOTHESIS','DIAGNOSIS','MEASURE_NEXT','EXPERIMENT_CANDIDATE','VERIFIED_IMPROVEMENT')}
    check('18 native observations exactly preserved',
          [f.to_dict() for f in automatic.findings if f.kind=='OBSERVATION'] == [f.to_dict() for f in old.findings if f.kind=='OBSERVATION'])
    check('Existing next-evidence findings retained',
          {f.finding_id for f in old.findings if f.kind=='MEASURE_NEXT'} <= {f.finding_id for f in automatic.findings if f.kind=='MEASURE_NEXT'})
    check('Real incomplete baseline has no runnable batch or precision candidate', counts(automatic)['EXPERIMENT_CANDIDATE']==0)
    check('No diagnosis, hypothesis or verified improvement', all(counts(automatic)[k]==0 for k in ('DIAGNOSIS','HYPOTHESIS','VERIFIED_IMPROVEMENT')))
    plans = [f for f in automatic.findings if type(f) is PlannedFinding]
    check('Every plan is blocked/not authorized with unknown cost/effect', all(
        (f.plan.authorization,f.plan.readiness,f.plan.cost,f.plan.expected_effect,f.plan.verified_effect)==
        ('not_authorized','blocked','unknown','unknown','not_measured') for f in plans))
    check('All candidate/review references remain source-eligible', all(
        resolver.check_metric(f.resolved_metric).status=='supported' for f in automatic.findings if type(f) is NativeObservation))
    observation = [f.resolved_metric for f in automatic.findings if type(f) is NativeObservation]
    check('P01 and P03 timing not interchanged', any(m.metric==M.FORWARD_P95 and m.value==.86681 for m in observation)
          and any(m.metric==M.E2E_P95 and m.value==4.642905 for m in observation))
    check('P02 remains batch32 images/s, not camera FPS', any(m.metric==M.COMPLETED_IMAGES_PER_SECOND and
        m.reference.unit=='images/s' and m.native_context['batch']==32 for m in observation))
    precision = [f for f in plans if f.rule_id=='plan.precision-quality']
    check('P08 BF16 not a speedup candidate; INT8 unsupported retained', bool(precision) and all(f.kind=='MEASURE_NEXT' for f in precision)
          and any('int8_dynamic:unsupported' in s for f in precision for s in f.plan.missing_evidence))
    check('Memory and quality guardrails explained', any(f.rule_id=='plan.memory-domain' for f in plans)
          and any(f.rule_id=='plan.accuracy-preservation' for f in plans))
    risks='\n'.join(automatic.risks)
    check('P18/P29/P31/B4 limits preserved', all(s in risks for s in (
        'qualified_stable_roof=not_established_by_this_report','independently_verified_improvement=not_certified',
        'long_run_7200_acceptance=not_certified','B4 native risk remains OPEN')))
    priorities = {}
    auto_facts = {f.finding_id:f.to_dict() for f in automatic.findings}
    for priority in ('speed','memory','preserve_accuracy','balanced'):
        variant = _advise_validated(resolver, AdvisorRequest(priority=priority), plan_experiments=True)
        check('Priority '+priority+' changes order only', {f.finding_id:f.to_dict() for f in variant.findings}==auto_facts
              and variant.risks==automatic.risks)
        priorities[priority] = next(f.plan.domain for f in variant.findings if type(f) is PlannedFinding)
    conflicting = _advise_validated(resolver, old_goals.request, plan_experiments=True)
    check('Prior GoalEvaluation results preserved', conflicting.goal_evaluations==old_goals.goal_evaluations)
    check('Hard-goal conflict blocks optimization proposals', conflicting.request_validation.consistency=='logically_inconsistent'
          and counts(conflicting)['EXPERIMENT_CANDIDATE']==0
          and all(f.rule_id=='plan.hard-goal-consistency' for f in conflicting.findings if type(f) is PlannedFinding))
    tough = next(g for g in old_goals.request.goals if g.target==.001)
    advanced = _advise_validated(resolver, AdvisorRequest([tough]), plan_experiments=True)
    check('Aggressive goal stays valid/missed, not impossible', advanced.goal_evaluations[0].input_validity=='valid'
          and advanced.goal_evaluations[0].observed=='observed_missed' and advanced.goal_evaluations[0].feasibility=='unknown')
    check('Missed latency goal receives evidence plans, not guaranteed optimization', any(
        f.rule_id=='plan.latency-stage' and tough.goal_id in f.plan.goal_ids for f in advanced.findings if type(f) is PlannedFinding))
    check('Same source/request yields deterministic findings and order',
          _advise_validated(resolver, AdvisorRequest(), plan_experiments=True).to_dict()==automatic.to_dict())
    args.output.mkdir(parents=True, exist_ok=False)
    for name,report in (('mnist-automatic',automatic),('mnist-advanced',advanced),('mnist-conflicting-goals',conflicting)):
        report.to_json(args.output/(name+'.json')); report.to_markdown(args.output/(name+'.md'))
        reopened=AdvisorReport.from_json(args.output/(name+'.json'))
        check(name+' JSON/Markdown roundtrip', reopened==report and reopened.to_markdown()==report.to_markdown())
    check('Planner derivation against actual source', validate_advisor_report(automatic,source))
    check('Old source/evidence bytes preserved', all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items()))
    check('No workload/factory/device calls', not calls and not torch.cuda.is_initialized())
    summary=dict(status='PASS',passed=len(checks),failed=0,checks=checks,
        automatic_counts=counts(automatic),advanced_counts=counts(advanced),conflicting_counts=counts(conflicting),
        previous_automatic_counts=counts(old),priority_first_plan_domain=priorities,
        source_report_id=resolver.source_report_id,planner_report_id=automatic.report_id,
        input_sha256=before,forbidden_calls=calls,cuda_initialized=False)
    (args.output/'acceptance.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
