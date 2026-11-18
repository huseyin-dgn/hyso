"""Goal evaluation against retained fixtures; no forward, training or device work."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import test_advisor_native as native_fixture
from hyso.analysis import AnalysisContext, EvidenceIndex, EvidenceRecord, EvidenceScope
import hashlib
from hyso.analysis._codec import digest
from hyso.advisor import (
    AdvisorError, AdvisorReport, AdvisorRequest, EvaluationProtocol, EvidenceSupport,
    Feasibility, Metric, MetricScope, NativeMetricResolver, ObservedGoal, UserGoal,
    advise, evaluate_goal, evaluate_goals, goal_context, validate_advisor_report, validate_request,
)
from hyso.advisor._json import canonical
from hyso.advisor.engine import _advise_validated


class GoalEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the retained-payload fixture and its forward/Trainer/CUDA guards.
        fixture = native_fixture.NativeAdvisorTests
        fixture.setUpClass()
        cls.addClassCleanup(fixture.doClassCleanups)
        # Goal completeness needs explicit fixture workload/preprocessing labels.
        # The older observation fixture deliberately omitted them. Supply caller
        # declarations only in this synthetic test envelope, never as native facts.
        records = []
        with tempfile.TemporaryDirectory() as directory:
            for feature, data in fixture.records.items():
                if data['native_payload'] is None:
                    continue
                raw = canonical(data['native_payload']).encode()
                path = Path(directory) / (feature + '.json'); path.write_bytes(raw)
                records.append(EvidenceRecord.from_json(path, feature_id=feature,
                    expected_sha256=hashlib.sha256(raw).hexdigest(), origin='reused',
                    scope=EvidenceScope(model_id='fixture-model', dataset_id='fixture-dataset',
                                        workload_id='fixture:' + feature, preprocessing_id='fixture-preprocessing')))
        cls.source = EvidenceIndex(records).analyze(context=AnalysisContext(
            workload_id='goal-fixture', task='classification', model={}, dataset={}))
        cls.resolver = NativeMetricResolver(cls.source)
        cls.records = {r['feature_id']: r for r in cls.source.to_dict()['index']['records']}
        cls.before = canonical(cls.source.to_dict())

    def goal(self, metric=Metric.RESIDENT_FORWARD_P95, feature='P01', **kwargs):
        context_metric = metric if metric in (Metric.RESIDENT_FORWARD_P95, Metric.E2E_P95,
            Metric.IMAGE_THROUGHPUT, Metric.ACCURACY, Metric.PROCESS_RSS_PEAK, Metric.LOGICAL_TENSOR_BYTES) else Metric.RESIDENT_FORWARD_P95
        scope, protocol = goal_context(self.resolver, self.records[feature]['record_id'], context_metric)
        values = dict(goal_id='goal', metric=metric, operator='<=', target=1.0, unit='ms', scope=scope, protocol=protocol)
        return UserGoal(**(values | kwargs))

    def test_normal_p95_observed_met_record_only(self):
        evaluation = evaluate_goal(self.resolver, self.goal())
        self.assertEqual((evaluation.input_validity, evaluation.evidence_support, evaluation.observed, evaluation.feasibility),
                         ('valid', 'supported', 'observed_met', 'unknown'))
        self.assertEqual(len(evaluation.evidence), 1)
        self.assertIn('deployment_acceptance:not_evaluated', ';'.join(evaluation.reasons))

    def test_aggressive_positive_goal_missed_not_invalid_or_impossible(self):
        g = self.goal(target=.001)
        evaluation = evaluate_goal(self.resolver, g)
        self.assertEqual((evaluation.input_validity, evaluation.observed, evaluation.feasibility),
                         ('valid', 'observed_missed', 'unknown'))
        self.assertNotIn('IMPOSSIBLE', ' '.join(evaluation.reasons))

    def test_e2e_p95_and_resident_not_mixed(self):
        g = self.goal(Metric.E2E_P95, 'P03', target=5)
        self.assertEqual(evaluate_goal(self.resolver, g).observed, 'observed_met')
        forward = self.goal(protocol=g.protocol)
        self.assertEqual(evaluate_goal(self.resolver, forward).observed, 'unknown')

    def test_p50_policy_cannot_masquerade_as_p95(self):
        g = self.goal()
        policy = json.loads(g.protocol.sample_policy)
        policy['native_metric'] = 'resident_forward_p50'
        altered = replace(g, protocol=replace(g.protocol, sample_policy=canonical(policy)))
        self.assertEqual(evaluate_goal(self.resolver, altered).observed, 'unknown')

    def test_accuracy_fraction_and_display_percentage(self):
        g = self.goal(Metric.ACCURACY, 'P26', operator='>=', target=.7, unit='fraction')
        evaluation = evaluate_goal(self.resolver, g)
        self.assertEqual(evaluation.observed, 'observed_met')
        result = advise(self.source, request=AdvisorRequest([g]))
        self.assertIn('0.75 fraction (75%)', result.to_markdown())
        for target, unit in ((75, 'fraction'), (75, 'percent')):
            with self.subTest(target=target, unit=unit), self.assertRaises(AdvisorError):
                replace(g, target=target, unit=unit)

    def test_rss_is_process_scope_not_cuda_or_inference_memory(self):
        g = self.goal(Metric.PROCESS_RSS_PEAK, 'P10', target=1, unit='GiB')
        evaluation = evaluate_goal(self.resolver, g)
        self.assertEqual(g.scope.phase, 'process')
        self.assertEqual((evaluation.observed, evaluation.evidence[0].domain), ('observed_met', 'process_rss'))
        bad_phase = replace(g, scope=replace(g.scope, phase='inference'))
        self.assertEqual(evaluate_goal(self.resolver, bad_phase).observed, 'unknown')
        vram = replace(g, metric=Metric.CUDA_ALLOCATED_PEAK)
        self.assertEqual(evaluate_goal(self.resolver, vram).evidence_support, 'unsupported_capability')

    def test_static_logical_footprint_not_runtime_ram(self):
        g = self.goal(Metric.LOGICAL_TENSOR_BYTES, 'P11', target=90000, unit='byte')
        evaluation = evaluate_goal(self.resolver, g)
        self.assertEqual(g.scope.phase, 'static')
        self.assertEqual((evaluation.observed, evaluation.evidence[0].domain), ('observed_met', 'model_logical_bytes'))
        self.assertEqual(evaluate_goal(self.resolver, replace(g, scope=replace(g.scope, phase='training'))).observed, 'unknown')

    def test_throughput_missing_backend_not_repaired(self):
        g = self.goal(Metric.IMAGE_THROUGHPUT, 'P02', target=10000, operator='>=', unit='images/s')
        self.assertIsNone(g.scope.backend)
        self.assertEqual(evaluate_goal(self.resolver, g).input_validity, 'needs_scope')
        asserted = replace(g, scope=replace(g.scope, backend='pytorch_eager'))
        evaluation = evaluate_goal(self.resolver, asserted)
        self.assertEqual(evaluation.observed, 'unknown')
        self.assertIn('scope_missing:backend', evaluation.reasons)

    def test_throughput_goal_with_explicit_native_backend_fixture(self):
        # Simulate an explicit producer field in a test envelope; do not modify
        # the accepted native fixture or claim this is new hardware evidence.
        payload = json.loads(canonical(self.records['P02']['native_payload']))
        payload['provenance']['environment']['backend'] = 'fixture-native-backend'
        with tempfile.TemporaryDirectory() as directory:
            raw = canonical(payload).encode(); path = Path(directory) / 'throughput.json'; path.write_bytes(raw)
            record = EvidenceRecord.from_json(path, feature_id='P02', expected_sha256=hashlib.sha256(raw).hexdigest(),
                scope=EvidenceScope(model_id='fixture-model', workload_id='fixture-throughput'))
        source = EvidenceIndex([record]).analyze(context=AnalysisContext(
            workload_id='throughput-fixture', task='classification', model={}, dataset={}))
        resolver = NativeMetricResolver(source)
        scope, protocol = goal_context(resolver, record.record_id, Metric.IMAGE_THROUGHPUT)
        g = UserGoal('rate', Metric.IMAGE_THROUGHPUT, '>=', 10000, 'images/s', scope=scope, protocol=protocol)
        self.assertEqual(evaluate_goal(resolver, g).observed, 'observed_met')
        self.assertEqual(evaluate_goal(resolver, replace(g, scope=replace(scope, batch=1, input_shape=(1, 1, 28, 28)))).observed, 'unknown')

    def test_images_and_camera_fps_not_equal(self):
        g = self.goal(metric=Metric.CAMERA_FPS, unit='frames/s')
        evaluation = evaluate_goal(self.resolver, g)
        self.assertEqual((evaluation.evidence_support, evaluation.observed), ('unsupported_capability', 'unknown'))

    def test_unsupported_energy_and_relative_baseline(self):
        g = self.goal(metric=Metric.ENERGY_PER_INFERENCE, unit='J/inference')
        self.assertEqual(evaluate_goal(self.resolver, g).evidence_support, 'unsupported_capability')
        g = self.goal(baseline_id='a' * 64)
        self.assertIn('baseline_comparison_requires_ComparisonContract', evaluate_goal(self.resolver, g).reasons)

    def test_estimated_goal_not_measured_result(self):
        self.assertEqual(evaluate_goal(self.resolver, self.goal(value_basis='estimated')).evidence_support, 'unsupported_capability')

    def test_declared_and_missing_provenance_stay_unproven(self):
        evaluation = evaluate_goal(self.resolver, self.goal())
        bindings = {b.field: b.authority for b in evaluation.evidence[0].bindings}
        self.assertEqual(bindings['model_id'], 'declared')
        self.assertEqual(bindings['precision'], 'missing')
        self.assertIn('future_feasibility:unknown', ';'.join(evaluation.reasons))

    def test_scope_mismatch_never_met(self):
        g = self.goal()
        variants = [dict(device='cuda:0'), dict(precision='fp64'), dict(workload_id='other'),
                    dict(dataset_id='other'), dict(model_id='other'), dict(clock_id='perf_counter_ns'),
                    dict(input_shape=(1, 3, 28, 28)), dict(batch=2, input_shape=(2, 1, 28, 28))]
        for kwargs in variants:
            with self.subTest(kwargs=kwargs):
                self.assertEqual(evaluate_goal(self.resolver, replace(g, scope=replace(g.scope, **kwargs))).observed, 'unknown')

    def test_missing_scope_not_inferred(self):
        g = self.goal(scope=None)
        evaluation = evaluate_goal(self.resolver, g)
        self.assertEqual((evaluation.input_validity, evaluation.observed), ('needs_scope', 'unknown'))
        self.assertIn('missing:scope', evaluation.reasons)

    def test_protocol_fields_require_exact_semantics(self):
        g = self.goal()
        for kwargs in (dict(protocol_id='serial_bounded_pipeline'), dict(version=2), dict(sample_policy='1000 samples'),
                       dict(window='camera_frame'), dict(evaluator='other'), dict(averaging='macro')):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(evaluate_goal(self.resolver, replace(g, protocol=replace(g.protocol, **kwargs))).observed, 'unknown')
        self.assertEqual(evaluate_goal(self.resolver, replace(g, protocol=replace(g.protocol, description='Display only'))).observed, 'observed_met')

    def test_exact_unit_conversion(self):
        self.assertEqual(evaluate_goal(self.resolver, self.goal(target=.001, unit='s')).observed, 'observed_met')
        g = self.goal(Metric.PROCESS_RSS_PEAK, 'P10', target=.6, unit='GB')
        self.assertEqual(evaluate_goal(self.resolver, g).observed, 'observed_missed')
        self.assertEqual(evaluate_goal(self.resolver, replace(g, unit='GiB')).observed, 'observed_met')

    def test_all_comparison_operators(self):
        for operator, expected in (('<=', 'observed_met'), ('>=', 'observed_met'), ('==', 'observed_met'),
                                   ('<', 'observed_missed'), ('>', 'observed_missed')):
            with self.subTest(operator=operator):
                self.assertEqual(evaluate_goal(self.resolver, self.goal(target=.86681, operator=operator)).observed, expected)

    def test_direct_hard_contradiction_remains_separate_axis(self):
        a = self.goal(goal_id='upper'); b = self.goal(goal_id='lower', operator='>=', target=2)
        request = AdvisorRequest([a, b])
        self.assertEqual(validate_request(request).consistency, 'logically_inconsistent')
        outcomes = evaluate_goals(self.resolver, request)
        self.assertEqual(tuple(g.observed for g in outcomes), ('observed_met', 'observed_missed'))
        self.assertTrue(all(g.input_validity == 'valid' for g in outcomes))

    def test_different_scopes_and_metrics_do_not_conflict(self):
        a = self.goal(goal_id='upper')
        b = self.goal(goal_id='lower', operator='>=', target=2)
        variants = [dict(scope=replace(b.scope, device='cuda:0')),
                    dict(scope=replace(b.scope, batch=2, input_shape=(2, 1, 28, 28))),
                    dict(metric=Metric.E2E_P95)]
        for kwargs in variants:
            with self.subTest(kwargs=kwargs):
                self.assertEqual(validate_request(AdvisorRequest([a, replace(b, **kwargs)])).consistency, 'no_conflict_identified')
        process = self.goal(Metric.PROCESS_RSS_PEAK, 'P10', unit='byte', target=1)
        training = replace(process, goal_id='training', operator='>=', target=2, scope=replace(process.scope, phase='training'))
        self.assertEqual(validate_request(AdvisorRequest([process, training])).consistency, 'no_conflict_identified')

    def test_high_throughput_low_ram_and_accuracy_drop_no_invented_impossibility(self):
        speed = self.goal(Metric.IMAGE_THROUGHPUT, 'P02', goal_id='speed', target=1e12, operator='>=', unit='images/s')
        memory = self.goal(Metric.PROCESS_RSS_PEAK, 'P10', goal_id='ram', target=1, unit='byte')
        quality = self.goal(metric=Metric.ACCURACY_DROP, scope=MetricScope('evaluation', workload_id='q', model_id='m',
            dataset_id='d', split='test', preprocessing_id='p'), goal_id='drop', target=0, unit='percentage_point', baseline_id='a' * 64)
        request = AdvisorRequest([speed, memory, quality])
        self.assertEqual(validate_request(request).consistency, 'no_conflict_identified')
        self.assertTrue(all(g.feasibility == Feasibility.UNKNOWN for g in evaluate_goals(self.resolver, request)))

    def test_priority_hard_soft_and_results_are_independent(self):
        hard = self.goal(goal_id='hard', target=.001)
        soft = self.goal(goal_id='soft', constraint='soft')
        request = AdvisorRequest([hard, soft])
        a = _advise_validated(self.resolver, request)
        b = _advise_validated(self.resolver, replace(request, priority='memory'))
        self.assertEqual(a.goal_evaluations, b.goal_evaluations)
        self.assertEqual(a.request.goals, b.request.goals)
        self.assertEqual({f.finding_id: f for f in a.findings}, {f.finding_id: f for f in b.findings})

    def test_ambiguous_records_not_cherry_picked(self):
        record = self.records['P01']
        duplicate = json.loads(canonical(record)); duplicate['origin'] = 'new'
        duplicate['record_id'] = digest({k: v for k, v in duplicate.items() if k != 'record_id'})
        source = EvidenceIndex([EvidenceRecord(record), EvidenceRecord(duplicate)]).analyze(
            context=AnalysisContext(
                workload_id='duplicate-fixture', task='classification', model={}, dataset={}))
        evaluation = evaluate_goal(NativeMetricResolver(source), self.goal())
        self.assertEqual((evaluation.evidence_support, evaluation.observed), ('insufficient_evidence', 'unknown'))
        self.assertEqual(len(evaluation.evidence), 2)

    def test_automatic_and_priority_legacy_data_unchanged(self):
        for request in (AdvisorRequest(), AdvisorRequest(priority='speed'), AdvisorRequest(priority='memory')):
            with self.subTest(request=request):
                current = _advise_validated(self.resolver, request)
                legacy = _advise_validated(self.resolver, request, evaluate_goals=False)
                self.assertEqual(current.to_dict(), legacy.to_dict())

    def test_old_advanced_observations_only_report_still_reproducible(self):
        old = _advise_validated(self.resolver, AdvisorRequest([self.goal()]), evaluate_goals=False)
        self.assertEqual(old.evaluation_status, 'observations_only')
        self.assertTrue(validate_advisor_report(old, self.source))

    def test_json_markdown_goal_roundtrip_and_derivation(self):
        result = _advise_validated(self.resolver, AdvisorRequest([self.goal()]))
        self.assertEqual(result.evaluation_status, 'goals_evaluated')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'goals.json'; result.to_json(path)
            reopened = AdvisorReport.from_json(path)
            self.assertEqual(reopened, result)
            text = reopened.to_markdown()
            self.assertIn('observed_met', text)
            self.assertIn('0.86681 ms', text)
            self.assertIn('deployment_acceptance:not_evaluated', text)
            self.assertTrue(validate_advisor_report(reopened, self.source))

    def test_fabricated_goal_outcome_not_valid_derivation(self):
        result = _advise_validated(self.resolver, AdvisorRequest([self.goal()]))
        fake = replace(result, goal_evaluations=(replace(result.goal_evaluations[0], observed='observed_missed'),))
        AdvisorReport.from_dict(fake.to_dict())
        with self.assertRaisesRegex(AdvisorError, 'derivation'):
            validate_advisor_report(fake, self.source)

    def test_goal_engine_does_not_produce_verified_or_future_guarantee(self):
        result = _advise_validated(self.resolver, AdvisorRequest([self.goal()]))
        self.assertFalse(any(f.kind == 'VERIFIED_IMPROVEMENT' for f in result.findings))
        with self.assertRaises(AdvisorError):
            replace(result, goal_evaluations=(replace(result.goal_evaluations[0], feasibility='demonstrated_in_scope'),))

    def test_original_scope_and_evidence_preserved(self):
        evaluate_goal(self.resolver, self.goal())
        self.assertEqual(canonical(self.source.to_dict()), self.before)
        self.assertIsNone(self.resolver.record(self.records['P01']['record_id'])['scope']['backend'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
