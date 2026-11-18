"""G1 pure planning contract. NO model, CUDA, video, process or sensor execution."""
import json
import unittest

from hyso.analysis.feature_plan import (
    CATALOG, SPECS, DEPENDENCIES, GOAL_FEATURES,
    FeaturePlanError, FeaturePlanRequest, ModelContext, plan_features,
)


def model(name='A', **kwargs):
    return ModelContext(name, input_contract=True, **kwargs)


def request(mode, models=None, **kwargs):
    return FeaturePlanRequest(mode, (model(),) if models is None else models, **kwargs)


def find(plan, feature):
    return next(row for row in plan.features if row.feature_id == feature)


class G1PlanningContract(unittest.TestCase):
    def test_all_35_canonical_ids_no_duplicates(self):
        self.assertEqual(tuple(SPECS), tuple(f'P{i:02d}' for i in range(1, 36)))
        self.assertEqual(len(CATALOG), 35)
        self.assertEqual(len(set(x.name for x in CATALOG)), 35)

    def test_full_mode_lists_every_feature_without_claiming_pass(self):
        result = plan_features(request('full'))
        self.assertEqual(len(result.features), 35)
        self.assertFalse(result.execution_authorized)
        self.assertEqual({x.authorization for x in result.features}, {'NOT_AUTHORIZED'})
        self.assertEqual({x.execution for x in result.features}, {'NOT_RUN'})
        self.assertEqual({x.acceptance for x in result.features}, {'NOT_EVALUATED'})

    def test_single_one_module(self):
        out = plan_features(request('single', features=('P01',)))
        self.assertEqual([x.feature_id for x in out.features], ['P01'])
        self.assertEqual(find(out, 'P01').models[0].status, 'PROPOSED')

    def test_selected_independent_modules(self):
        out = plan_features(request('selected', features=('P10', 'P01', 'P02')))
        self.assertEqual([x.feature_id for x in out.features], ['P01', 'P02', 'P10'])
        self.assertEqual({x.selection for x in out.features}, {'requested'})

    def test_selected_dependency_is_visible_not_executed(self):
        out = plan_features(request('single', features=('P29',)))
        self.assertEqual([x.feature_id for x in out.features],
                         ['P01', 'P02', 'P09', 'P10', 'P26', 'P29'])
        self.assertEqual(find(out, 'P29').selection, 'requested')
        self.assertEqual(find(out, 'P26').selection, 'prerequisite')
        self.assertEqual(find(out, 'P26').execution, 'NOT_RUN')
        self.assertEqual(find(out, 'P29').models[0].status, 'BLOCKED')

    def test_automatic_goal_selection_without_authorization(self):
        out = plan_features(request('automatic', goals=('latency', 'throughput')))
        self.assertIn('P01', [x.feature_id for x in out.features])
        self.assertIn('P17', [x.feature_id for x in out.features])
        self.assertNotIn('P30', [x.feature_id for x in out.features])
        self.assertEqual(find(out, 'P01').selection, 'recommended')
        self.assertFalse(out.execution_authorized)

    def test_two_models_never_certifies_a_winner(self):
        out = plan_features(request('single', models=(model('old'), model('new')),
                                    features=('P01',)))
        row = find(out, 'P01')
        self.assertEqual(tuple(x.model_id for x in row.models), ('old', 'new'))
        self.assertEqual(row.comparison, 'PENDING_COMPATIBILITY_EVIDENCE')
        self.assertEqual(out.comparison_authority, 'NO_NATIVE_COMPARISON_OR_CERTIFIED_WINNER')

    def test_two_models_with_one_missing_input_blocks_comparison(self):
        out = plan_features(request('selected', models=(model('old'), ModelContext('new')),
                                    features=('P01', 'P02')))
        self.assertEqual(find(out, 'P01').comparison, 'BLOCKED')
        self.assertEqual(find(out, 'P02').models[1].status, 'BLOCKED')

    def test_gpu_requires_native_runtime_verification(self):
        for cap in ((), ('gpu',)):
            with self.subTest(cap=cap):
                out = plan_features(request('single', features=('P09',), capabilities=cap))
                row = find(out, 'P09')
                self.assertEqual(row.models[0].status, 'BLOCKED')
                reason = ' '.join(row.models[0].reasons)
                if cap:
                    self.assertIn('runtime_capability_requires_native_verification:gpu', reason)
                else:
                    self.assertIn('missing_declared_capability:gpu', reason)

    def test_max_batch_requires_resource_policy_not_just_factory(self):
        out = plan_features(request('single', models=(model(trusted_factory=True),),
                                    features=('P17',)))
        self.assertIn('bounded_worker_resource_policy_not_attested',
                      find(out, 'P17').models[0].reasons)

    def test_roofline_reference_cannot_be_promoted_to_target(self):
        out = plan_features(request('single', features=('P18',),
                                    capabilities=('gpu', 'ncu')))
        self.assertIn('target_model_kernel_counter_binding_not_attested',
                      find(out, 'P18').models[0].reasons)

    def test_long_run_and_memory_are_never_silently_certified(self):
        out = plan_features(request('selected', models=(model(retained_long_run=True, trusted_factory=True),),
                                    features=('P31', 'P32')))
        self.assertEqual(find(out, 'P31').models[0].status, 'BLOCKED')
        self.assertIn('uninterrupted_7200_second_run_requires_separate_authorization',
                      find(out, 'P31').models[0].reasons)
        self.assertIn('retained_7200_source_requires_identity_and_continuity_validation',
                      find(out, 'P32').models[0].reasons)

    def test_camera_is_not_inferred_from_mnist_or_checkpoint(self):
        out = plan_features(request('single', features=('P30',)))
        reasons = find(out, 'P30').models[0].reasons
        self.assertIn('missing_task_appropriate_trained_frame_adapter', reasons)
        self.assertIn('missing_recorded_video_or_live_camera', reasons)

    def test_energy_requires_real_matched_power_data(self):
        out = plan_features(request('single', features=('P33',), capabilities=('gpu', 'power_sensor')))
        self.assertIn('matching_idle_active_model_power_windows_not_attested',
                      find(out, 'P33').models[0].reasons)
        self.assertEqual(find(out, 'P16').selection, 'prerequisite')

    def test_training_not_assumed_from_inference(self):
        out = plan_features(request('automatic', goals=('training',)))
        self.assertEqual(find(out, 'P19').models[0].status, 'BLOCKED')
        self.assertIn('missing_training_workload', find(out, 'P19').models[0].reasons)

    def test_detection_classification_metric_not_silently_reused(self):
        ctx = model(task='detection', adapter_contract=True, labeled_evaluation=True)
        out = plan_features(request('single', models=(ctx,), features=('P26',)))
        self.assertEqual(find(out, 'P26').models[0].status, 'BLOCKED')
        self.assertIn('classification_metrics_not_a_substitute_for_task_specific_quality',
                      find(out, 'P26').models[0].reasons)

    def test_nonclassification_requires_task_adapter(self):
        ctx = model(task='segmentation')
        out = plan_features(request('single', models=(ctx,), features=('P01',)))
        self.assertIn('non_classification_model_requires_task_specific_adapter',
                      find(out, 'P01').models[0].reasons)

    def test_onnx_inference_producer_never_falsely_available(self):
        ctx = model(backend='onnx')
        out = plan_features(request('selected', models=(ctx,), features=('P01', 'P35'),
                                    capabilities=('ort_cpu',)))
        self.assertEqual(find(out, 'P01').models[0].status, 'NOT_APPLICABLE')
        self.assertEqual(find(out, 'P35').models[0].status, 'BLOCKED')

    def test_python_model_needs_explicit_onnx_export(self):
        out = plan_features(request('single', features=('P35',), capabilities=('ort_cpu',)))
        self.assertIn('missing_declared_onnx_export_contract', find(out, 'P35').models[0].reasons)

    def test_constrained_serialization_is_deterministic(self):
        req = request('selected', models=(model('A'), model('B')),
                      features=('P01', 'P29'), capabilities=('gpu',))
        a = plan_features(req).to_dict()
        b = plan_features(req).to_dict()
        self.assertEqual(a, b)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        self.assertEqual(a['execution_authorized'], False)

    def test_bad_mode_and_user_provided_ids_rejected(self):
        for kwargs in (
            dict(mode='execute', models=(model(),)),
            dict(mode='single', models=(model(),), features=('P00',)),
            dict(mode='single', models=(model(),), features=('P01', 'P02')),
            dict(mode='automatic', models=(model(),), goals=()),
            dict(mode='selected', models=(model(),), features=('P01', 'P01')),
            dict(mode='selected', models=(model(),), features=('P01',), capabilities=('CUDA',)),
            dict(mode='full', models=(model(),), features=('P01',)),
            dict(mode='selected', models=(model(),), features=('P01',), goals=('latency',)),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(FeaturePlanError):
                FeaturePlanRequest(**kwargs)

    def test_bad_models_rejected(self):
        for models in ((), (model(), model('B'), model('C')),
                       (model('X'), model('X')), (object(),)):
            with self.subTest(models=models), self.assertRaises(FeaturePlanError):
                FeaturePlanRequest('full', models)

    def test_bad_boolean_and_ids_rejected(self):
        with self.assertRaises(FeaturePlanError):
            ModelContext("x", input_contract=1)
        with self.assertRaises(FeaturePlanError):
            ModelContext('x', trusted_factory=1)
        with self.assertRaises(FeaturePlanError):
            ModelContext(' ')

    def test_unknown_input_object_rejected(self):
        with self.assertRaises(FeaturePlanError):
            plan_features(object())

    def test_dependencies_are_canonical_and_acyclic(self):
        ids = set(SPECS)
        self.assertTrue(all(x in ids and set(parents) <= ids for x, parents in DEPENDENCIES.items()))
        for f in ids:
            seen = set()
            def visit(node, path):
                if node in path:
                    self.fail('cyclic dependency')
                for p in DEPENDENCIES.get(node, ()): visit(p, path | {node})
            visit(f, seen)
        self.assertTrue(all(set(v) <= ids for v in GOAL_FEATURES.values()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
