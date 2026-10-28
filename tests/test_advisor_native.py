"""Focused read-only resolver/eligibility tests: no model forward or benchmark."""
from contextlib import ExitStack
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from hyso.analysis import AnalysisContext, EvidenceIndex, EvidenceRecord, EvidenceScope, UnifiedAnalysisReport
from hyso.advisor import (AdvisorError, AdvisorReport, AdvisorRequest, EvidenceBinding,
                          EvidenceSupport, FindingKind, Metric, MetricResolution,
                          NativeMetric as M, NativeMetricResolver, ResolvedMetric, UserGoal,
                          advise, validate_advisor_report)
from hyso.advisor._json import canonical


class NativeAdvisorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert not torch.cuda.is_initialized()
        cls.guards = ExitStack()
        def forbidden(*args, **kwargs):
            raise AssertionError('Workload/device operation forbidden in read-only tests')
        for name in ('init', 'is_available', 'device_count', 'current_device'):
            cls.guards.enter_context(patch.object(torch.cuda, name, forbidden))
        cls.guards.enter_context(patch.object(torch.nn.Module, '_call_impl', forbidden))
        from hyso.benchmark import BenchmarkRunner
        from hyso.training import Trainer
        cls.guards.enter_context(patch.object(BenchmarkRunner, 'run', forbidden))
        cls.guards.enter_context(patch.object(Trainer, 'fit', forbidden))
        cls.addClassCleanup(cls.guards.close)
        payloads = json.loads((Path(__file__).parent / 'fixtures/advisor-native-v1.json').read_text(encoding='utf-8'))['payloads']
        # Confusion arithmetic over fixed labels, not model inference or a measurement.
        from hyso import evaluate_predictions
        payloads['P26'] = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 1],
                                               class_names=['a', 'b'], split='test').to_dict()
        cls.artifacts = {}; records = []
        with tempfile.TemporaryDirectory() as directory:
            for feature, payload in payloads.items():
                raw = canonical(payload).encode('utf-8')
                path = Path(directory) / (feature + '.json'); path.write_bytes(raw)
                record = EvidenceRecord.from_json(path, feature_id=feature,
                    expected_sha256=hashlib.sha256(raw).hexdigest(), origin='reused',
                    scope=EvidenceScope(model_id='fixture-model', dataset_id='fixture-dataset'))
                records.append(record); cls.artifacts[record.record_id] = raw
        records.append(EvidenceRecord.unavailable('P09', status='unsupported', reason='No CUDA measurement'))
        cls.source = EvidenceIndex(records).analyze(context=AnalysisContext(
            workload_id='fixture-analysis', task='image_classification', model={}, dataset={}))
        cls.before = canonical(cls.source.to_dict())
        cls.resolver = NativeMetricResolver(cls.source)
        cls.records = {r['feature_id']: r for r in cls.source.to_dict()['index']['records']}
        cls.result = advise(cls.source)

    def resolved(self, feature, metric, **kwargs):
        result = self.resolver.resolve(self.records[feature]['record_id'], metric, **kwargs)
        self.assertEqual(result.status, 'supported', result.reasons)
        return result.resolved

    def point(self, precision):
        return next(p for p in self.records['P08']['native_payload']['raw_evidence']['points'] if p['precision'] == precision)['point_id']

    def test_quantile_identity_and_exact_p95(self):
        m = self.resolved('P01', M.FORWARD_P95)
        self.assertEqual(m.value, .86681)
        self.assertEqual(m.native_context['quantile'], .95)
        self.assertEqual(m.reference.metric_path, 'measurements.1.value')
        self.assertEqual(m.native_context['measurement_scope'], 'synchronous_host_forward')
        self.assertEqual(m.native_context['sample_count'], 1000)

    def test_p50_p99_are_separate_identities(self):
        self.assertEqual(self.resolved('P01', M.FORWARD_P50).value, .77945)
        self.assertEqual(self.resolved('P01', M.FORWARD_P99).value, .997323)

    def test_semantic_native_scope_guard(self):
        # The old native quantile constructor does not whitelist timing scope;
        # the resolver must independently reject an invented scope label.
        payload = json.loads(canonical(self.records['P01']['native_payload']))
        payload['protocol']['scope'] = 'current_stream_forward'
        with tempfile.TemporaryDirectory() as directory:
            raw = canonical(payload).encode()
            path = Path(directory) / 'scope.json'; path.write_bytes(raw)
            record = EvidenceRecord.from_json(path, feature_id='P01', expected_sha256=hashlib.sha256(raw).hexdigest())
        source = EvidenceIndex([record]).analyze(context=AnalysisContext(
            workload_id='scope-fixture', task='classification', model={}, dataset={}))
        result = NativeMetricResolver(source).resolve(record.record_id, M.FORWARD_P95)
        self.assertEqual(result.status, 'insufficient_evidence')
        self.assertIn('resident_forward_protocol_not_supported', result.reasons)

    def test_semantic_estimated_reference_and_unknown_metric(self):
        m = self.resolved('P01', M.FORWARD_P95)
        ref = replace(m.reference, status='estimated')
        self.assertEqual(self.resolver.check_reference(ref, m.metric).status, EvidenceSupport.INSUFFICIENT_EVIDENCE)
        unknown = self.resolver.resolve(m.reference.record_id, 'undefined_metric')
        self.assertEqual(unknown.status, 'unsupported_metric')
        self.assertIsNone(unknown.resolved)

    def test_semantic_structured_observation_roundtrip(self):
        from hyso.advisor import NativeObservation
        observation = next(f for f in self.result.findings if type(f) is NativeObservation)
        reopened = NativeObservation.from_dict(observation.to_dict())
        self.assertEqual(reopened.resolved_metric.value, observation.resolved_metric.value)
        self.assertEqual(reopened.resolved_metric.record_scope, observation.resolved_metric.record_scope)
        self.assertEqual(self.resolver.check_metric(reopened.resolved_metric).status, EvidenceSupport.SUPPORTED)

    def test_fake_record_id_rejected(self):
        ref = replace(self.resolved('P01', M.FORWARD_P95).reference, record_id='f' * 64)
        self.assertEqual(self.resolver.check_reference(ref, M.FORWARD_P95).status, EvidenceSupport.INSUFFICIENT_EVIDENCE)

    def test_wrong_hashes_rejected(self):
        ref = self.resolved('P01', M.FORWARD_P95).reference
        for field in ('artifact_sha256', 'payload_sha256', 'report_id'):
            with self.subTest(field=field):
                wrong = replace(ref, **{field: 'f' * 64})
                self.assertEqual(self.resolver.check_reference(wrong, M.FORWARD_P95).status, EvidenceSupport.INSUFFICIENT_EVIDENCE)

    def test_wrong_path_and_p50_as_p95_rejected(self):
        ref = self.resolved('P01', M.FORWARD_P95).reference
        for path in ('measurements.0.value', 'measurements.999.value'):
            with self.subTest(path=path):
                self.assertEqual(self.resolver.check_reference(replace(ref, metric_path=path), M.FORWARD_P95).status,
                                 EvidenceSupport.INSUFFICIENT_EVIDENCE)
        p50 = self.resolved('P01', M.FORWARD_P50)
        self.assertEqual(self.resolver.check_reference(p50.reference, M.FORWARD_P95).status, EvidenceSupport.INSUFFICIENT_EVIDENCE)

    def test_wrong_unit_domain_status_rejected(self):
        ref = self.resolved('P02', M.COMPLETED_IMAGES_PER_SECOND).reference
        for field, value in (('unit', 'frames/s'), ('domain', 'camera_fps'), ('status', 'estimated')):
            with self.subTest(field=field):
                self.assertEqual(self.resolver.check_reference(replace(ref, **{field: value}), M.COMPLETED_IMAGES_PER_SECOND).status,
                                 EvidenceSupport.INSUFFICIENT_EVIDENCE)

    def test_completed_images_not_camera_fps(self):
        m = self.resolved('P02', M.COMPLETED_IMAGES_PER_SECOND)
        self.assertAlmostEqual(m.value, 12442.170637760393)
        self.assertEqual((m.reference.domain, m.reference.unit, m.native_context['batch']), ('throughput', 'images/s', 32))
        self.assertEqual(self.resolver.resolve(m.reference.record_id, 'camera_fps').status, 'unsupported_metric')

    def test_e2e_not_resident_forward(self):
        e2e = self.resolved('P03', M.E2E_P95)
        self.assertEqual((e2e.value, e2e.reference.domain), (4.642905, 'e2e'))
        self.assertEqual(self.resolver.resolve(e2e.reference.record_id, M.FORWARD_P95).status, 'unsupported_metric')
        forward = self.resolved('P01', M.FORWARD_P95)
        result = self.resolver.compare(forward.reference, forward.metric, e2e.reference, e2e.metric)
        self.assertEqual(result.status, EvidenceSupport.INSUFFICIENT_EVIDENCE)
        self.assertIn('metric_semantics_incompatible', result.reasons)

    def test_insufficient_tail_not_zero(self):
        result = self.resolver.resolve(self.records['P03']['record_id'], M.E2E_P99)
        self.assertEqual(result.status, 'insufficient_evidence')
        self.assertIsNone(result.resolved)

    def test_sweep_requires_exact_point(self):
        rid = self.records['P08']['record_id']
        for point in (None, 'invented-point'):
            with self.subTest(point=point):
                self.assertEqual(self.resolver.resolve(rid, M.PRECISION_MEAN, point_id=point).status, 'insufficient_evidence')

    def test_precision_context_preserved_without_promoting_record(self):
        m = self.resolved('P08', M.PRECISION_MEAN, point_id=self.point('bf16'))
        self.assertEqual((m.value, m.native_context['precision'], m.native_context['batch']), (1.618805, 'bf16', 1))
        self.assertIsNone(m.record_scope['precision'])
        self.assertEqual(dict((b.field, b.authority) for b in m.reference.bindings)['precision'], 'missing')
        self.assertEqual(m.protocol['precision'], 'bf16')  # Configuration default fp32 is not the point precision.

    def test_sweep_throughput_uses_native_domain(self):
        m = self.resolved('P08', M.PRECISION_THROUGHPUT, point_id=self.point('fp32'))
        projection = next(x for x in self.records['P08']['metrics'] if x['path'] == m.reference.metric_path)
        self.assertEqual(projection['domain'], 'latency')
        self.assertEqual(m.reference.domain, 'throughput')
        self.assertEqual(m.reference.unit, 'images/s')
        self.assertTrue(any('Projection domain disagrees' in v for v in m.limitations))

    def test_unsupported_int8_not_zero_or_recommendation(self):
        result = self.resolver.resolve(self.records['P08']['record_id'], M.PRECISION_MEAN, point_id=self.point('int8_dynamic'))
        self.assertEqual(result.status, 'insufficient_evidence')
        self.assertIsNone(result.resolved)

    def test_cross_point_reference_scope_rejected(self):
        m = self.resolved('P08', M.PRECISION_MEAN, point_id=self.point('bf16'))
        wrong = replace(m.reference, point_ref=self.point('fp32'))
        self.assertEqual(self.resolver.check_reference(wrong, m.metric).status, EvidenceSupport.INSUFFICIENT_EVIDENCE)
        context = m.native_context | {'precision': 'fp32', 'batch': 32}
        self.assertEqual(self.resolver.check_metric(replace(m, native_context_json=canonical(context))).status,
                         EvidenceSupport.INSUFFICIENT_EVIDENCE)

    def test_rss_cuda_logical_and_artifact_separate(self):
        rss = self.resolved('P10', M.RSS_SAMPLED_PEAK)
        logical = self.resolved('P11', M.MODEL_LOGICAL_BYTES)
        artifact = self.resolved('P11', M.ARTIFACT_BYTES)
        self.assertEqual((rss.value, logical.value, artifact.value), (605245440, 81048, 101532))
        self.assertEqual((rss.reference.domain, logical.reference.domain, artifact.reference.domain),
                         ('process_rss', 'model_logical_bytes', 'serialized_artifact_bytes'))
        self.assertEqual(self.resolver.check_reference(replace(rss.reference, domain='cuda_allocator'), rss.metric).status,
                         EvidenceSupport.INSUFFICIENT_EVIDENCE)

    def test_unsupported_physical_memory(self):
        result = self.resolver.resolve(self.records['P11']['record_id'], 'physical_memory_bytes')
        self.assertEqual(result.status, 'unsupported_metric')
        self.assertIsNone(result.resolved)

    def test_test_accuracy_and_macro_f1_fraction(self):
        accuracy = self.resolved('P26', M.TEST_ACCURACY)
        self.assertEqual((accuracy.value, accuracy.reference.unit, accuracy.native_context['split']), (.75, 'fraction', 'test'))
        self.assertEqual(self.resolved('P26', M.TEST_MACRO_F1).native_context['averaging'], 'macro')

    def test_declared_native_escalation_rejected(self):
        ref = self.resolved('P01', M.FORWARD_P95).reference
        bindings = tuple(EvidenceBinding(b.field, 'native' if b.field == 'model_id' else b.authority) for b in ref.bindings)
        self.assertEqual(self.resolver.check_reference(replace(ref, bindings=bindings), M.FORWARD_P95).status,
                         EvidenceSupport.INSUFFICIENT_EVIDENCE)

    def test_observation_does_not_require_all_native_provenance(self):
        m = self.resolved('P01', M.FORWARD_P95)
        self.assertEqual(self.resolver.check_reference(m.reference, m.metric).status, EvidenceSupport.SUPPORTED)
        self.assertEqual(self.resolver.check_reference(m.reference, m.metric, required_native=['model_id']).status,
                         EvidenceSupport.INSUFFICIENT_EVIDENCE)

    def test_same_declarations_do_not_certify_comparison(self):
        m = self.resolved('P01', M.FORWARD_P95)
        result = self.resolver.compare(m.reference, m.metric, m.reference, m.metric)
        self.assertEqual(result.status, EvidenceSupport.INSUFFICIENT_EVIDENCE)
        self.assertTrue(any('native_association_' in v for v in result.reasons))

    def test_explicit_artifact_bytes_checked(self):
        m = self.resolved('P01', M.FORWARD_P95)
        self.assertEqual(self.resolver.check_reference(m.reference, m.metric, artifact_bytes=self.artifacts[m.reference.record_id]).status,
                         EvidenceSupport.SUPPORTED)
        self.assertIn('artifact_sha256_mismatch', self.resolver.check_reference(m.reference, m.metric, artifact_bytes=b'{}').reasons)

    def test_unavailable_record_has_no_value(self):
        result = self.resolver.resolve(self.records['P09']['record_id'], M.RSS_CURRENT)
        self.assertEqual(result.status, 'insufficient_evidence')
        self.assertIsNone(result.resolved)

    def test_resolved_value_origin_protocol_tamper(self):
        m = self.resolved('P01', M.FORWARD_P95)
        for kwargs in ({'value': 999}, {'origin': 'new'}, {'protocol_json': canonical({'id': 'fake'})}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.resolver.check_metric(replace(m, **kwargs)).status, EvidenceSupport.INSUFFICIENT_EVIDENCE)

    def test_tampered_analysis_report_rejected(self):
        for mutate in (lambda d: d['index']['records'][0]['native_payload']['measurements'][0].update(value=999),
                       lambda d: d.update(schema_version=2), lambda d: d.update(report_id='f' * 64)):
            data = self.source.to_dict(); mutate(data)
            # A typed but bypass-constructed object must also be revalidated at the boundary.
            forged = object.__new__(UnifiedAnalysisReport)
            object.__setattr__(forged, '_payload', canonical(data))
            with self.subTest(mutate=mutate), self.assertRaisesRegex(AdvisorError, 'invalid_analysis_report'):
                NativeMetricResolver(forged)

    def test_rehashed_native_corruption_rejected(self):
        data = self.source.to_dict()
        r = data['index']['records'][0]
        r['native_payload']['measurements'][0]['value'] = 999
        r['record_id'] = hashlib.sha256(canonical({k: v for k, v in r.items() if k != 'record_id'}).encode()).hexdigest()
        forged = object.__new__(UnifiedAnalysisReport); object.__setattr__(forged, '_payload', canonical(data))
        with self.assertRaises(AdvisorError):
            NativeMetricResolver(forged)

    def test_only_observations_and_conditional_evidence_actions(self):
        self.assertTrue(self.result.findings)
        self.assertTrue(all(f.kind in (FindingKind.OBSERVATION, FindingKind.MEASURE_NEXT) for f in self.result.findings))
        self.assertFalse(any(f.verification == 'verified' for f in self.result.findings))
        self.assertEqual(self.result.request.goals, ())
        self.assertEqual(self.result.evaluation_status, 'planning_only')

    def test_priority_changes_order_only(self):
        variants = [advise(self.source, request=AdvisorRequest(priority=p))
                    for p in ('speed', 'memory', 'preserve_accuracy', 'balanced')]
        base = {f.finding_id: f.to_dict() for f in self.result.findings}
        for variant in variants:
            self.assertEqual({f.finding_id: f.to_dict() for f in variant.findings}, base)
            self.assertEqual(variant.risks, self.result.risks)
            self.assertEqual(variant.evidence_status, self.result.evidence_status)
        self.assertTrue(variants[1].findings[0].statement.startswith('P10'))
        self.assertTrue(variants[2].findings[0].statement.startswith('P26'))

    def test_incomplete_advanced_goal_remains_unknown(self):
        request = AdvisorRequest(goals=[UserGoal('hard', Metric.RESIDENT_FORWARD_P95, '<=', 1e-20, 'ms')], priority='speed')
        result = advise(self.source, request=request)
        self.assertEqual(result.request, request)
        self.assertEqual(result.goal_evaluations[0].evidence_support, 'insufficient_evidence')
        self.assertEqual(result.goal_evaluations[0].observed, 'unknown')
        self.assertEqual(result.goal_evaluations[0].feasibility, 'unknown')

    def test_risk_boundaries_preserved(self):
        text = '\n'.join(self.result.risks)
        for expected in ('B4 native risk remains OPEN', 'not_established_by_this_report', 'not_certified', 'not_proven'):
            self.assertIn(expected, text)

    def test_json_markdown_roundtrip_no_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path, md_path = Path(directory) / 'advisor.json', Path(directory) / 'advisor.md'
            self.result.to_json(json_path); self.result.to_markdown(md_path)
            reopened = AdvisorReport.from_json(json_path)
            self.assertEqual(reopened, self.result)
            self.assertEqual(reopened.to_markdown(), self.result.to_markdown())
            self.assertIn('0.86681 ms', md_path.read_text(encoding='utf-8'))
            with self.assertRaises(FileExistsError):
                self.result.to_markdown(md_path)
        m = self.resolved('P01', M.FORWARD_P95)
        self.assertEqual(ResolvedMetric.from_dict(m.to_dict()), m)
        self.assertEqual(MetricResolution.from_dict(MetricResolution('supported', m).to_dict()).resolved, m)

    def test_derivation_checked_against_source_not_hash_shape(self):
        self.assertTrue(validate_advisor_report(self.result, self.source))
        findings = list(self.result.findings)
        findings[0] = replace(findings[0], statement='invented observation')
        forged = replace(self.result, findings=tuple(findings))
        # This has valid contract hashes, but is not the engine's derivation.
        AdvisorReport.from_dict(forged.to_dict())
        with self.assertRaisesRegex(AdvisorError, 'derivation'):
            validate_advisor_report(forged, self.source)

    def test_immutable_snapshots_and_no_source_changes(self):
        m = self.resolved('P01', M.FORWARD_P95)
        scope = m.record_scope; scope['batch'] = 999
        self.assertEqual(m.record_scope['batch'], 1)
        with self.assertRaises(FrozenInstanceError):
            self.resolver.source_report_id = 'fake'
        self.assertEqual(canonical(self.source.to_dict()), self.before)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main(verbosity=2)
