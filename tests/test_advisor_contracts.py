"""Focused 3B.1 CPU contracts; run directly with unittest, without pytest fixtures."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hyso.advisor import (
    AdvisorError, AdvisorReport, AdvisorRequest, Consistency, EvaluationProtocol,
    EvidenceBinding, EvidenceRef, EvidenceSupport, Feasibility, Finding, FindingKind,
    GoalEvaluation, InputValidity, Metric, MetricScope, ObservedGoal, Priority,
    RequestValidation, ResourceBudget, RuleVersion, UserGoal, validate_request,
)
from hyso.advisor._json import MAX_BYTES, canonical


SHA = 'a' * 64
SCOPE = MetricScope('inference', device='cpu', backend='torch', precision='fp32',
                    batch=1, input_shape=(1, 1, 28, 28), workload_id='resident', model_id='model')
PROTOCOL = EvaluationProtocol('resident-tail', 1, 'Host forward P95', sample_policy='1000 samples')


def goal(**kwargs):
    values = dict(goal_id='latency', metric=Metric.RESIDENT_FORWARD_P95,
                  operator='<=', target=1.0, unit='ms', scope=SCOPE, protocol=PROTOCOL)
    return UserGoal(**(values | kwargs))


def reference(**kwargs):
    values = dict(report_id=SHA, record_id='b' * 64, metric_path='measurements.1.value',
                  artifact_sha256='c' * 64, payload_sha256='d' * 64, domain='latency',
                  unit='ms', status='derived', bindings=(EvidenceBinding('model_id', 'declared'),))
    return EvidenceRef(**(values | kwargs))


def rehash(data):
    body = {k: v for k, v in data.items() if k != 'content_sha256'}
    data['content_sha256'] = hashlib.sha256(canonical(body).encode()).hexdigest()
    return data


class RequestTests(unittest.TestCase):
    def test_automatic_has_no_invented_target(self):
        request = AdvisorRequest()
        self.assertEqual((request.mode, request.goals, request.budgets, request.priority),
                         ('automatic', (), (), None))

    def test_all_priorities_only_change_priority(self):
        for priority in Priority:
            with self.subTest(priority=priority):
                request = AdvisorRequest(priority=priority)
                self.assertEqual(request.mode, 'priority')
                self.assertEqual(request.goals, ())
                self.assertEqual(request.budgets, ())

    def test_advanced_and_priority_hard_soft_independent(self):
        request = AdvisorRequest([goal(), goal(goal_id='soft', constraint='soft')], priority='memory')
        self.assertEqual(request.mode, 'advanced')
        self.assertEqual([g.constraint for g in request.goals], ['hard', 'soft'])
        self.assertEqual(request.priority, Priority.MEMORY)

    def test_aggressive_goal_remains_valid(self):
        self.assertEqual(goal(target=1e-12).input_validity, InputValidity.VALID)

    def test_missing_scope_not_filled(self):
        g = goal(scope=None, protocol=None)
        self.assertEqual(g.input_validity, InputValidity.NEEDS_SCOPE)
        self.assertIsNone(g.scope)
        self.assertIsNone(g.protocol)
        evaluation = validate_request(AdvisorRequest([g])).goals[0]
        self.assertEqual(evaluation.reasons, ('missing:scope', 'missing:protocol'))

    def test_partial_scope_not_promoted(self):
        g = goal(scope=MetricScope('inference'))
        self.assertEqual(g.input_validity, InputValidity.NEEDS_SCOPE)
        self.assertIn('scope.precision', g.missing_scope_fields)

    def test_phase_separation(self):
        with self.assertRaisesRegex(AdvisorError, 'phase'):
            goal(scope=replace(SCOPE, phase='training'))

    def test_negative_latency_rejected(self):
        with self.assertRaises(AdvisorError):
            goal(target=-1)

    def test_nonfinite_targets_rejected(self):
        for value in (float('nan'), float('inf'), float('-inf'), True, '1', 10**1000):
            with self.subTest(value=str(value)[:20]), self.assertRaises(AdvisorError):
                goal(target=value)

    def test_accuracy_fraction_range_and_unit(self):
        for value, unit in ((1.01, 'fraction'), (98.9, 'fraction'), (98.9, 'percent')):
            with self.subTest(value=value, unit=unit), self.assertRaises(AdvisorError):
                goal(metric=Metric.ACCURACY, target=value, unit=unit, scope=None)
        g = goal(metric=Metric.ACCURACY, target=.989, unit='fraction', scope=None)
        self.assertEqual(g.target, .989)

    def test_drop_requires_reference_and_percentage_points(self):
        with self.assertRaisesRegex(AdvisorError, 'baseline'):
            goal(metric=Metric.ACCURACY_DROP, unit='percentage_point', scope=None)
        g = goal(metric=Metric.ACCURACY_DROP, unit='percentage_point', baseline_id=SHA, scope=None)
        self.assertEqual(g.unit, 'percentage_point')

    def test_images_and_camera_frames_not_interchangeable(self):
        for metric, unit in ((Metric.CAMERA_FPS, 'images/s'), (Metric.IMAGE_THROUGHPUT, 'frames/s')):
            with self.subTest(metric=metric), self.assertRaisesRegex(AdvisorError, 'unit'):
                goal(metric=metric, unit=unit)
        self.assertNotEqual(goal(metric=Metric.CAMERA_FPS, unit='frames/s').metric,
                            goal(metric=Metric.IMAGE_THROUGHPUT, unit='images/s').metric)

    def test_metric_identity_is_closed(self):
        for metric in ('latency', 'memory', 'user.factory'):
            with self.subTest(metric=metric), self.assertRaises(AdvisorError):
                goal(metric=metric)

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(AdvisorError, 'duplicate'):
            AdvisorRequest([goal(), goal()])

    def test_scope_shape_and_types_rejected(self):
        for kwargs in ({'batch': True}, {'input_shape': [1, False]}, {'batch': 2}, {'device': ''}):
            with self.subTest(kwargs=kwargs), self.assertRaises(AdvisorError):
                replace(SCOPE, **kwargs)


class StaticConsistencyTests(unittest.TestCase):
    def test_direct_hard_p95_contradiction(self):
        result = validate_request(AdvisorRequest([goal(), goal(goal_id='lower', operator='>=', target=2)]))
        self.assertEqual(result.consistency, Consistency.INCONSISTENT)
        self.assertEqual(result.conflicting_goal_ids, ('latency', 'lower'))

    def test_static_result_embedded_and_revalidated(self):
        request = AdvisorRequest([goal(), goal(goal_id='lower', operator='>=', target=2)])
        result = validate_request(request)
        report = AdvisorReport(SHA, SHA, request, request_validation=result)
        self.assertEqual(AdvisorReport.from_dict(report.to_dict()).request_validation, result)
        with self.assertRaises(AdvisorError):
            AdvisorReport(SHA, SHA, request, request_validation=validate_request(AdvisorRequest()))

    def test_strict_boundary_contradiction(self):
        result = validate_request(AdvisorRequest([goal(operator='<'), goal(goal_id='lower', operator='>=')]))
        self.assertEqual(result.consistency, Consistency.INCONSISTENT)

    def test_equal_inclusive_boundary_allowed(self):
        result = validate_request(AdvisorRequest([goal(), goal(goal_id='lower', operator='>=')]))
        self.assertEqual(result.consistency, Consistency.NO_CONFLICT)

    def test_equality_and_mixed_units(self):
        result = validate_request(AdvisorRequest([goal(operator='=='),
            goal(goal_id='lower', operator='>=', target=.002, unit='s')]))
        self.assertEqual(result.consistency, Consistency.INCONSISTENT)

    def test_gb_gib_use_exact_distinct_scales(self):
        a = goal(metric=Metric.PROCESS_RSS_PEAK, target=1, unit='GB')
        b = replace(a, goal_id='lower', target=1, unit='GiB', operator='>=')
        self.assertEqual(validate_request(AdvisorRequest([a, b])).consistency, Consistency.INCONSISTENT)

    def test_soft_conflict_not_hard_inconsistency(self):
        result = validate_request(AdvisorRequest([goal(), goal(goal_id='soft', target=2, operator='>=', constraint='soft')]))
        self.assertEqual(result.consistency, Consistency.NO_CONFLICT)

    def test_different_contexts_not_combined(self):
        variants = [dict(scope=replace(SCOPE, batch=2, input_shape=(2, 1, 28, 28))),
                    dict(protocol=replace(PROTOCOL, version=2)), dict(value_basis='estimated'),
                    dict(metric=Metric.E2E_P95), dict(scope=None)]
        for kwargs in variants:
            with self.subTest(kwargs=kwargs):
                b = goal(goal_id='lower', target=2, operator='>=', **kwargs)
                self.assertEqual(validate_request(AdvisorRequest([goal(), b])).consistency, Consistency.NO_CONFLICT)

    def test_high_throughput_low_memory_no_invented_conflict(self):
        request = AdvisorRequest([goal(metric=Metric.IMAGE_THROUGHPUT, unit='images/s', operator='>=', target=1e9),
                                  goal(goal_id='ram', metric=Metric.PROCESS_RSS_PEAK, unit='byte', target=1)])
        result = validate_request(request)
        self.assertEqual(result.consistency, Consistency.NO_CONFLICT)
        self.assertTrue(all(e.feasibility == Feasibility.UNKNOWN for e in result.goals))
        self.assertTrue(all(e.observed == ObservedGoal.UNKNOWN for e in result.goals))
        self.assertTrue(all(e.evidence_support is None for e in result.goals))

    def test_rss_cuda_and_logical_domains_not_combined(self):
        for metric in (Metric.CUDA_ALLOCATED_PEAK, Metric.CUDA_RESERVED_PEAK, Metric.LOGICAL_TENSOR_BYTES):
            with self.subTest(metric=metric):
                a = goal(metric=Metric.PROCESS_RSS_PEAK, unit='byte', target=1)
                b = goal(goal_id='other', metric=metric, unit='byte', operator='>=', target=1000)
                self.assertEqual(validate_request(AdvisorRequest([a, b])).consistency, Consistency.NO_CONFLICT)


class BudgetTests(unittest.TestCase):
    def budget(self, **kwargs):
        return ResourceBudget(**(dict(budget_id='ram', purpose='experiment', domain='process_rss', limit=1, unit='GiB') | kwargs))

    def test_positive_finite_budget_only(self):
        for value in (-1, 0, float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(AdvisorError):
                self.budget(limit=value)

    def test_budget_domain_and_units(self):
        ram = self.budget()
        vram = self.budget(domain='cuda_allocated')
        self.assertNotEqual(ram.domain, vram.domain)
        self.assertNotEqual(self.budget(unit='GB').content_sha256, ram.content_sha256)
        with self.assertRaises(AdvisorError):
            self.budget(unit='ms')

    def test_budget_does_not_claim_enforcement(self):
        budget = self.budget(allowed_devices=['cpu'], allowed_precisions=['fp32'])
        self.assertEqual(budget.enforcement, 'not_evaluated')
        self.assertEqual(budget.applicability, 'not_evaluated')
        with self.assertRaises(AdvisorError):
            self.budget(enforcement='enforced')

    def test_bad_purpose_window_and_trial_count(self):
        for kwargs in ({'purpose': 'training'}, {'window_seconds': 0},
                       {'domain': 'trial_count', 'unit': 'count', 'limit': 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(AdvisorError):
                self.budget(**kwargs)


class EvidenceAndReportTests(unittest.TestCase):
    def test_reference_is_syntax_only_and_declared_stays_declared(self):
        ref = reference()  # These hashes need not exist on disk.
        self.assertEqual(EvidenceRef.from_dict(ref.to_dict()).bindings[0].authority, 'declared')

    def test_bad_reference_hash_path_and_binding(self):
        for kwargs in ({'artifact_sha256': 'x'}, {'metric_path': '__import__("factory")'},
                       {'bindings': [EvidenceBinding('a', 'native'), EvidenceBinding('a', 'missing')]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(AdvisorError):
                reference(**kwargs)
        with self.assertRaises(AdvisorError):
            EvidenceBinding('a', 'verified')

    def test_verified_label_cannot_be_fabricated(self):
        with self.assertRaisesRegex(AdvisorError, 'comparison_contract_unavailable'):
            Finding('fake', FindingKind.VERIFIED_IMPROVEMENT, 'Claim', evidence=[reference()])
        with self.assertRaises(AdvisorError):
            Finding('fake', FindingKind.OBSERVATION, 'Claim', verification='verified')

    def test_other_finding_kinds_roundtrip(self):
        for kind in FindingKind:
            if kind != FindingKind.VERIFIED_IMPROVEMENT:
                with self.subTest(kind=kind):
                    f = Finding('f', kind, 'Caller supplied statement', evidence=[reference()])
                    self.assertEqual(Finding.from_dict(f.to_dict()), f)
                    self.assertEqual(f.verification, 'not_evaluated')

    def test_unknown_not_zero_and_no_default_support(self):
        e = validate_request(AdvisorRequest([goal()])).goals[0]
        self.assertEqual((e.observed, e.feasibility, e.evidence_support),
                         (ObservedGoal.UNKNOWN, Feasibility.UNKNOWN, None))
        report = AdvisorReport(SHA, 'e' * 64, AdvisorRequest())
        self.assertEqual(report.evaluation_status, 'not_evaluated')
        self.assertIsNone(report.evidence_status)
        self.assertEqual(report.findings, ())

    def test_supported_and_observed_require_explicit_references(self):
        with self.assertRaises(AdvisorError):
            GoalEvaluation('g', 'valid', evidence_support='supported')
        with self.assertRaises(AdvisorError):
            GoalEvaluation('g', 'valid', observed='observed_met', evidence=[reference()])

    def test_independent_axes_roundtrip_without_claiming_validation(self):
        e = GoalEvaluation('latency', 'valid', evidence_support='supported', observed='observed_met',
                           feasibility='demonstrated_in_scope', evidence=[reference()])
        report = AdvisorReport(SHA, SHA, AdvisorRequest([goal()]), goal_evaluations=[e],
                               evidence_status='supported', evaluation_status='caller_supplied')
        self.assertEqual(report.assessment_authority, 'caller_supplied_not_verified')
        self.assertEqual(AdvisorReport.from_dict(report.to_dict()), report)

    def test_report_rejects_wrong_goal_source_and_rule(self):
        request = AdvisorRequest([goal()])
        bad = [dict(goal_evaluations=[GoalEvaluation('other', 'valid')]),
               dict(findings=[Finding('f', 'OBSERVATION', 'x', evidence=[reference(report_id='f' * 64)])]),
               dict(findings=[Finding('f', 'OBSERVATION', 'x', rule_id='rule', rule_version='2')],
                    rule_versions=[RuleVersion('rule', '1')]),
               dict(evidence_status='supported')]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(AdvisorError):
                AdvisorReport(SHA, SHA, request, **kwargs)

    def test_invalid_axis_rejected(self):
        with self.assertRaises(AdvisorError):
            GoalEvaluation('g', 'valid', observed='PASS')


class SerializationTests(unittest.TestCase):
    def test_full_json_roundtrip_and_detached_data(self):
        ref = reference(point_ref='point-0', sample_ref='samples-0', protocol_ref='protocol-1')
        finding = Finding('f', 'OBSERVATION', 'Caller statement', evidence=[ref],
                          rule_id='caller-rule', rule_version='1', limitations=['Unresolved association'])
        request = AdvisorRequest([goal()], [ResourceBudget('b', 'deployment', 'process_rss', 1, 'GB')], 'speed')
        report = AdvisorReport(SHA, 'e' * 64, request, findings=[finding],
            goal_evaluations=validate_request(request).goals, rule_versions=[RuleVersion('caller-rule', '1')],
            risks=['B4 OPEN'], evidence_limitations=['No evidence resolved'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            report.to_json(path)
            self.assertEqual(AdvisorReport.from_json(path), report)
            with self.assertRaises(FileExistsError):
                report.to_json(path)
        data = report.to_dict()
        data['request']['goals'][0]['unit'] = 's'
        self.assertEqual(report.request.goals[0].unit, 'ms')
        self.assertEqual(report.request_sha256, request.content_sha256)
        self.assertEqual(report.request_id, request.request_id)

    def test_immutable_nested_inputs(self):
        goals = [goal()]
        shape = [1, 2]
        scope = MetricScope('inference', input_shape=shape)
        request = AdvisorRequest(goals)
        goals.clear()
        shape[0] = 99
        self.assertEqual(len(request.goals), 1)
        self.assertEqual(scope.input_shape, (1, 2))
        for obj, name, value in ((request, 'priority', 'speed'), (scope, 'phase', 'training'),
                                 (request.goals[0], 'target', 0), (reference().bindings[0], 'authority', 'native')):
            with self.subTest(name=name), self.assertRaises(FrozenInstanceError):
                setattr(obj, name, value)

    def test_all_contract_types_roundtrip(self):
        objects = [SCOPE, PROTOCOL, goal(), ResourceBudget('b', 'experiment', 'wall_time', 1, 's'),
                   AdvisorRequest(), reference(), EvidenceBinding('x', 'missing'),
                   GoalEvaluation('invalid-input', 'invalid', reasons=['Rejected before request construction']),
                   RequestValidation((), 'potentially_conflicting'), RuleVersion('rule', '1')]
        for obj in objects:
            with self.subTest(contract=type(obj).__name__):
                self.assertEqual(type(obj).from_dict(obj.to_dict()), obj)

    def test_unknown_schema_and_wrong_root_type(self):
        for version in (2, True, '1'):
            with self.subTest(version=version), self.assertRaisesRegex(AdvisorError, 'schema_version'):
                AdvisorRequest.from_dict(rehash(AdvisorRequest().to_dict() | {'schema_version': version}))
        with self.assertRaises(AdvisorError):
            AdvisorRequest.from_dict(goal().to_dict())

    def test_unknown_fields_and_tamper(self):
        with self.assertRaisesRegex(AdvisorError, 'fields'):
            AdvisorRequest.from_dict(AdvisorRequest().to_dict() | {'extra': 1})
        with self.assertRaisesRegex(AdvisorError, 'integrity'):
            AdvisorRequest.from_dict(AdvisorRequest().to_dict() | {'priority': 'speed'})
        report = AdvisorReport(SHA, SHA, AdvisorRequest()).to_dict()
        with self.assertRaisesRegex(AdvisorError, 'derived'):
            AdvisorReport.from_dict(rehash(report | {'request_id': 'forged'}))

    def test_rehashed_invalid_payload_still_validated(self):
        with self.assertRaises(AdvisorError):
            UserGoal.from_dict(rehash(goal().to_dict() | {'target': -1}))

    def test_json_duplicates_nonfinite_utf8_and_malformed(self):
        cases = [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'\xff', b'{']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            for raw in cases:
                with self.subTest(raw=raw), self.assertRaises(AdvisorError):
                    path.write_bytes(raw)
                    AdvisorRequest.from_json(path)

    def test_serialization_size_and_depth_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'large.json'
            path.write_bytes(b' ' * (MAX_BYTES + 1))
            with self.assertRaisesRegex(AdvisorError, 'size'):
                AdvisorRequest.from_json(path)
        data = AdvisorRequest().to_dict()
        data['priority'] = []
        for _ in range(30):
            data['priority'] = [data['priority']]
        with self.assertRaises(AdvisorError):
            AdvisorRequest.from_dict(rehash(data))
        with self.assertRaises(AdvisorError):
            AdvisorRequest(goals=[goal(goal_id=str(i)) for i in range(1025)])


class CompatibilityTests(unittest.TestCase):
    def test_existing_analysis_contract_and_fail_closed_comparison(self):
        from hyso.analysis import AnalysisContext, EvidenceIndex, EvidenceRecord, UnifiedAnalysisReport, compare_evidence
        index = EvidenceIndex([EvidenceRecord.unavailable('P01', reason='not measured')])
        context = AnalysisContext(workload_id='contract-only', task='classification', model={}, dataset={})
        report = index.analyze(context=context)
        before = report.to_dict()
        AdvisorReport(report.to_dict()['report_id'], hashlib.sha256(canonical(before).encode()).hexdigest(), AdvisorRequest())
        self.assertEqual(report.to_dict(), before)
        self.assertEqual(before['analysis']['recommendations'], {'status': 'not_implemented', 'items': []})
        self.assertEqual(UnifiedAnalysisReport(before).to_dict(), before)
        self.assertEqual(compare_evidence(index.records[0], index.records[0], domain='latency')['status'], 'incompatible')

    def test_no_runtime_factory_or_cuda_work_on_contract_import(self):
        # Isolated import audit, not a workload. sys.modules is observed without
        # importing a source factory or querying a device.
        script = """
import sys
import torch
assert not torch.cuda.is_initialized()
def deny(label):
    def blocked(*args, **kwargs):
        raise AssertionError('runtime work forbidden: ' + label)
    return blocked
# Distinct sentinels preserve Torch Dynamo's import-time object-rule registry.
torch.cuda.init = deny('cuda.init')
torch.cuda.is_available = deny('cuda.is_available')
torch.cuda.device_count = deny('cuda.device_count')
torch.cuda.current_device = deny('cuda.current_device')
def audit(event, args):
    if event == 'import' and args[0].startswith('untrusted_model_factories'):
        raise AssertionError('factory import forbidden')
sys.addaudithook(audit)
from hyso.advisor import AdvisorRequest, Metric, MetricScope, UserGoal, validate_request
import hyso.advisor as advisor
from hyso.benchmark import BenchmarkRunner
from hyso.training import Trainer
BenchmarkRunner.run = deny('benchmark')
Trainer.fit = deny('training')
before = set(sys.modules)
validate_request(AdvisorRequest(priority='speed', goals=[UserGoal(
    'g', Metric.RESIDENT_FORWARD_P95, '<=', 1, 'ms',
    scope=MetricScope('inference', model_id='untrusted_model_factories:build'),
)]))
assert callable(advisor.advise)  # Added in 3B.2; it must still perform no work on import.
assert not any(name.startswith('untrusted_model_factories') for name in set(sys.modules) - before)
assert not torch.cuda.is_initialized()
print('contract/import compatibility PASS')
"""
        result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
