"""Read-only conditional plan fixtures; no workloads or device interrogation."""
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

import test_advisor_goals as goal_fixture
from hyso.analysis import AnalysisContext, EvidenceIndex, EvidenceRecord, EvidenceScope
from hyso.advisor import (AdvisorError, AdvisorReport, AdvisorRequest, DecisionPlan, Metric,
                         NativeMetricResolver, NativeObservation, PlannedFinding, ResourceBudget,
                         UserGoal, advise, goal_context, validate_advisor_report)
from hyso.advisor._json import canonical
from hyso.advisor.engine import _advise_validated
from hyso.advisor.planning import _add_plans


class PlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f = goal_fixture.GoalEngineTests
        f.setUpClass(); cls.addClassCleanup(f.doClassCleanups)
        cls.source, cls.resolver, cls.records = f.source, f.resolver, f.records
        cls.baseline = _advise_validated(cls.resolver, AdvisorRequest())
        cls.result = _add_plans(cls.resolver, cls.baseline)

    def throughput_source(self, *, backend='fixture-native-backend', missing_model=False):
        # Test-only metadata variants over retained samples; not new measurements.
        import json
        payload = json.loads(canonical(self.records['P02']['native_payload']))
        if backend is None:
            payload['provenance']['environment'].pop('backend', None)
        else:
            payload['provenance']['environment']['backend'] = backend
        if missing_model:
            payload['provenance']['model'].pop('weights_sha256', None)
            payload['provenance']['model'].pop('state_sha256', None)
        with tempfile.TemporaryDirectory() as directory:
            raw = canonical(payload).encode(); path = Path(directory)/'throughput.json'; path.write_bytes(raw)
            record = EvidenceRecord.from_json(path, feature_id='P02', expected_sha256=hashlib.sha256(raw).hexdigest(), origin='reused',
                scope=EvidenceScope(model_id='fixture-model', workload_id='fixture-throughput'))
        source = EvidenceIndex([record]).analyze(context=AnalysisContext(workload_id='planner-fixture', task='classification', model={}, dataset={}))
        return source, record.record_id

    def planned(self, result, kind=None):
        return [f for f in result.findings if type(f) is PlannedFinding and (kind is None or f.kind == kind)]

    def test_positive_conditional_batch_candidate(self):
        source, rid = self.throughput_source()
        result = advise(source)
        candidates = self.planned(result, 'EXPERIMENT_CANDIDATE')
        self.assertEqual(len(candidates), 1)
        c = candidates[0]
        self.assertEqual(c.evidence[0].record_id, rid)
        self.assertEqual((c.plan.change_parameter, c.plan.changed_variables), ('batch_size', ('input.batch_size',)))
        self.assertEqual((c.plan.authorization, c.plan.readiness, c.plan.expected_effect, c.plan.cost),
                         ('not_authorized', 'blocked', 'unknown', 'unknown'))
        self.assertTrue(c.plan.fixed_conditions and c.plan.remeasure_metrics and c.plan.acceptance_conditions)

    def test_missing_provenance_or_capability_produces_review(self):
        for kwargs in ({'backend': None}, {'missing_model': True}):
            with self.subTest(kwargs=kwargs):
                source, _ = self.throughput_source(**kwargs); result = advise(source)
                self.assertEqual(self.planned(result, 'EXPERIMENT_CANDIDATE'), [])
                self.assertTrue(any(f.rule_id == 'plan.batch-throughput' and f.kind == 'MEASURE_NEXT' for f in self.planned(result)))

    def test_no_quality_memory_acceptance_or_budget_is_invented(self):
        source, _ = self.throughput_source()
        c = self.planned(advise(source), 'EXPERIMENT_CANDIDATE')[0]
        self.assertTrue(any('memory and quality' in s for s in c.plan.missing_evidence))
        self.assertTrue(any('budget' in s for s in c.plan.missing_evidence))
        self.assertEqual(c.plan.readiness, 'blocked')

    def test_p06_review_precedes_candidate_without_fabricated_sweep(self):
        source, _ = self.throughput_source(); result = advise(source)
        plans = self.planned(result)
        self.assertEqual(plans[0].plan.stage, 'retained_review')
        self.assertEqual(plans[-1].kind, 'EXPERIMENT_CANDIDATE')
        self.assertTrue(any('P06' in f.statement and f.plan.stage == 'retained_review' for f in plans))

    def test_latency_is_review_not_diagnosis_or_p01_p03_difference(self):
        p = next(f for f in self.planned(self.result) if f.rule_id == 'plan.latency-stage')
        self.assertEqual(p.kind, 'MEASURE_NEXT')
        self.assertTrue(any('do not subtract' in s for s in p.plan.fixed_conditions))
        self.assertFalse(any(f.kind in ('DIAGNOSIS', 'HYPOTHESIS') for f in self.result.findings))

    def test_bf16_and_unsupported_int8_not_candidates(self):
        plans = self.planned(self.result)
        precision = next(f for f in plans if f.rule_id == 'plan.precision-quality')
        self.assertEqual(precision.kind, 'MEASURE_NEXT')
        self.assertTrue(any('int8_dynamic:unsupported' in s for s in precision.plan.missing_evidence))
        self.assertFalse(any(f.kind == 'EXPERIMENT_CANDIDATE' and f.plan.domain == 'precision' for f in plans))

    def test_memory_domains_not_subtracted_or_renamed(self):
        p = next(f for f in self.planned(self.result) if f.rule_id == 'plan.memory-domain')
        self.assertEqual({r.domain for r in p.evidence}, {'process_rss', 'model_logical_bytes'})
        self.assertTrue(any('never subtract' in s for s in p.plan.fixed_conditions))
        self.assertIsNone(p.plan.change_parameter)

    def test_preserve_accuracy_has_no_numeric_tolerance(self):
        result = advise(self.source, request=AdvisorRequest(priority='preserve_accuracy'))
        p = next(f for f in self.planned(result) if f.rule_id == 'plan.accuracy-preservation')
        self.assertEqual(result.request.goals, ())
        self.assertIn('priority does not create a numerical accuracy tolerance', ' '.join(p.plan.fixed_conditions))

    def test_deployment_budget_does_not_authorize_or_limit_experiment(self):
        source, _ = self.throughput_source()
        budget = ResourceBudget('ram', 'deployment', 'process_rss', 1, 'GiB')
        c = self.planned(advise(source, request=AdvisorRequest(budgets=[budget])), 'EXPERIMENT_CANDIDATE')[0]
        self.assertEqual(c.plan.budget_ids, ('ram',))
        self.assertEqual(c.plan.authorization, 'not_authorized')
        self.assertTrue(any('Deployment budgets are not experiment limits' in s for s in c.plan.risks))

    def test_missed_throughput_goal_links_candidate_and_hard_acceptance(self):
        source, rid = self.throughput_source(); resolver = NativeMetricResolver(source)
        scope, protocol = goal_context(resolver, rid, Metric.IMAGE_THROUGHPUT)
        goal = UserGoal('rate', Metric.IMAGE_THROUGHPUT, '>=', 1e12, 'images/s', scope=scope, protocol=protocol)
        result = advise(source, request=AdvisorRequest([goal]))
        self.assertEqual(result.goal_evaluations[0].input_validity, 'valid')
        self.assertEqual(result.goal_evaluations[0].observed, 'observed_missed')
        c = self.planned(result, 'EXPERIMENT_CANDIDATE')[0]
        self.assertEqual(c.plan.goal_ids, ('rate',))
        self.assertTrue(any('Hard goal rate' in s for s in c.plan.acceptance_conditions))
        self.assertEqual(c.plan.expected_effect, 'unknown')

    def test_met_goal_creates_no_new_optimization_objective(self):
        source, rid = self.throughput_source(); resolver = NativeMetricResolver(source)
        scope, protocol = goal_context(resolver, rid, Metric.IMAGE_THROUGHPUT)
        g = UserGoal('rate', Metric.IMAGE_THROUGHPUT, '>=', 1, 'images/s', scope=scope, protocol=protocol)
        result = advise(source, request=AdvisorRequest([g]))
        self.assertEqual(result.goal_evaluations[0].observed, 'observed_met')
        self.assertEqual(self.planned(result), [])
        self.assertEqual(result.request.goals, (g,))

    def test_unknown_goal_has_no_wrong_candidate(self):
        source, _ = self.throughput_source()
        g = UserGoal('unknown', Metric.IMAGE_THROUGHPUT, '>=', 1e12, 'images/s')
        result = advise(source, request=AdvisorRequest([g]))
        self.assertEqual(result.goal_evaluations[0].observed, 'unknown')
        self.assertEqual(self.planned(result, 'EXPERIMENT_CANDIDATE'), [])

    def test_conflicting_hard_goals_block_candidate(self):
        source, rid = self.throughput_source(); resolver = NativeMetricResolver(source)
        scope, protocol = goal_context(resolver, rid, Metric.IMAGE_THROUGHPUT)
        a = UserGoal('lower', Metric.IMAGE_THROUGHPUT, '>=', 20000, 'images/s', scope=scope, protocol=protocol)
        b = replace(a, goal_id='upper', operator='<=', target=1000)
        result = advise(source, request=AdvisorRequest([a, b]))
        self.assertEqual(result.request_validation.consistency, 'logically_inconsistent')
        self.assertEqual(self.planned(result, 'EXPERIMENT_CANDIDATE'), [])
        self.assertEqual({f.rule_id for f in self.planned(result)}, {'plan.hard-goal-consistency'})

    def test_model_batch_precision_scope_mismatch_does_not_plan(self):
        source, rid = self.throughput_source(); resolver = NativeMetricResolver(source)
        scope, protocol = goal_context(resolver, rid, Metric.IMAGE_THROUGHPUT)
        goal = UserGoal('rate', Metric.IMAGE_THROUGHPUT, '>=', 1e12, 'images/s', scope=scope, protocol=protocol)
        for kwargs in (dict(model_id='other-model'), dict(precision='fp64'), dict(batch=1, input_shape=(1, 1, 28, 28))):
            with self.subTest(kwargs=kwargs):
                result = advise(source, request=AdvisorRequest([replace(goal, scope=replace(scope, **kwargs))]))
                self.assertEqual(result.goal_evaluations[0].observed, 'unknown')
                self.assertEqual(self.planned(result, 'EXPERIMENT_CANDIDATE'), [])

    def test_wrong_domain_or_gpu_claim_is_rejected_before_planning(self):
        from hyso.advisor import EvidenceRef
        target = next(f for f in self.baseline.findings if type(f) is NativeObservation)
        for ref in (replace(target.resolved_metric.reference, domain='cuda_allocator'),
                    replace(target.resolved_metric.reference, unit='frames/s')):
            m = replace(target.resolved_metric, reference=ref)
            f = replace(target, evidence=(ref,), resolved_metric=m)
            base = replace(self.baseline, findings=(f,))
            with self.assertRaises(AdvisorError):
                _add_plans(self.resolver, base)
        m = replace(target.resolved_metric, native_context_json=canonical(target.resolved_metric.native_context | {'device':'cuda:0'}))
        with self.assertRaises(AdvisorError):
            _add_plans(self.resolver, replace(self.baseline, findings=(replace(target, resolved_metric=m),)))

    def test_fake_effect_authorization_and_diagnosis_refused(self):
        source, _ = self.throughput_source(); c = self.planned(advise(source), 'EXPERIMENT_CANDIDATE')[0]
        for kwargs in (dict(expected_effect='2x'), dict(verified_effect='verified'), dict(cost='1 second'),
                       dict(authorization='authorized'), dict(readiness='ready'), dict(fixed_conditions=())):
            with self.subTest(kwargs=kwargs), self.assertRaises(AdvisorError):
                replace(c.plan, **kwargs)
        for kind in ('DIAGNOSIS', 'VERIFIED_IMPROVEMENT', 'HYPOTHESIS'):
            with self.subTest(kind=kind), self.assertRaises(AdvisorError):
                replace(c, kind=kind)

    def test_priority_changes_order_only_and_dependency_order_is_kept(self):
        base = {f.finding_id:f.to_dict() for f in self.result.findings}
        for priority in ('speed', 'memory', 'preserve_accuracy', 'balanced'):
            result = _advise_validated(self.resolver, AdvisorRequest(priority=priority), plan_experiments=True)
            self.assertEqual({f.finding_id:f.to_dict() for f in result.findings}, base)
            self.assertEqual(result.risks, self.result.risks)
            stages = [f.plan.stage for f in self.planned(result)]
            order = {'retained_review':0, 'association_validation':1, 'new_measurement':2, 'conditional_experiment':3}
            self.assertEqual([order[s] for s in stages], sorted(order[s] for s in stages))

    def test_observations_goals_and_determinism_preserved(self):
        base = [f for f in self.baseline.findings if f.kind == 'OBSERVATION']
        self.assertEqual([f for f in self.result.findings if f.kind == 'OBSERVATION'], base)
        self.assertEqual(self.result.goal_evaluations, self.baseline.goal_evaluations)
        self.assertEqual(_add_plans(self.resolver, self.baseline).to_dict(), self.result.to_dict())
        self.assertTrue(all(f.kind not in ('VERIFIED_IMPROVEMENT', 'DIAGNOSIS') for f in self.result.findings))
        self.assertTrue(any('7200_acceptance=not_certified' in s for s in self.result.risks))
        self.assertTrue(any('B4' in s and 'OPEN' in s for s in self.result.risks))

    def test_json_markdown_and_derivation(self):
        source, _ = self.throughput_source(); result = advise(source)
        reopened = AdvisorReport.from_dict(result.to_dict())
        self.assertEqual(reopened.to_markdown(), result.to_markdown())
        self.assertIn('Nothing is authorized', reopened.to_markdown())
        self.assertTrue(validate_advisor_report(reopened, source))
        candidate = self.planned(result, 'EXPERIMENT_CANDIDATE')[0]
        altered = replace(candidate, statement='Fake guaranteed optimization')
        forged = replace(result, findings=tuple(altered if f.finding_id == candidate.finding_id else f for f in result.findings))
        with self.assertRaisesRegex(AdvisorError, 'derivation'):
            validate_advisor_report(forged, source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
