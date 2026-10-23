"""13 targeted stdlib/protocol tests; no Torch import, CUDA, or process launch.

Load the real result schema and the real protocol dataclass in isolated package
stubs. Only the public API call is a spy; no inference implementation is run.
"""
import ast
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/forensics'))
import gpu_scenario_runner as runner
import native_scenario_runner as legacy
from offline_test_isolation import offline_test


@contextmanager
def public_schema():
    """Production schema/protocol code, with Torch-bearing package init omitted."""
    packages = {}
    for name in ('hyso', 'hyso.benchmark', 'hyso.performance'):
        package = types.ModuleType(name)
        package.__path__ = []
        packages[name] = package
    with patch.dict(sys.modules, packages):
        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
        names = ('hyso.benchmark.result', 'hyso.performance.result', 'hyso.performance.inference')
        with patch.dict(sys.modules):
            load(names[0], ROOT/'hyso/benchmark/result.py')
            result = load(names[1], ROOT/'hyso/performance/result.py')
            inference = types.ModuleType(names[2])
            sys.modules[names[2]] = inference
            inference.__dict__.update(dataclass=dataclass, math=math,
                _integer=result._integer, _positive=result._positive)
            tree = ast.parse((ROOT/'hyso/performance/inference.py').read_text())
            protocol = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='ThroughputProtocol')
            api = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='measure_throughput'))
            api.body = [ast.Return(ast.Name('_reply', ast.Load()))]
            compiled = ast.fix_missing_locations(ast.Module(body=[protocol, api], type_ignores=[]))
            exec(compile(compiled, 'production_protocol_and_api_signature', 'exec'), inference.__dict__)
            yield inference, result


def payload():
    row = dict(start_ns=1000000000, admission_end_ns=1004000000, end_ns=1005000000,
        admitted_batches=5, completed_batches=5, completed_images=10, pending_batches=0,
        completed_id_range=[0,4], completion_groups=3, max_pending_batches=2,
        backpressure_syncs=2, terminal_drains=1, elapsed_wall_seconds=.005,
        tail_seconds=.001, images_per_second=2000., repeat_index=0)
    return dict(format='hyso.performance.throughput', schema_version=1,
        result_id='11111111-1111-4111-8111-111111111111', feature_ids=['P02'],
        identity=dict(experiment_run_id=None, source_result_id=None),
        protocol=dict(id='continuous_completion_window', version=1,
            scope='resident_forward_completion_wall', configuration=dict(runner.P02_PROTOCOL),
            clock='perf_counter_ns', synchronization='bounded_group_completion_and_terminal_event',
            included=['forward','finite_check_kernels','loop_and_queue_bookkeeping','backpressure','terminal_tail'],
            excluded=['input_clone_transfer','preflight','warmup','state_checks','hashing','export','tracking']),
        provenance=dict(model={'qualified_class':'torch.nn.modules.conv.Conv2d'},
            input=dict(shape=[2,1,7,9], dtype='torch.float32', device='cuda:0', source='resident_clone'),
            environment=dict(device='cuda:0', stream_id=0)),
        requested=dict(batch_count=5, duration_seconds=None, repeats=1, warmup_iterations=1),
        actual=dict(repeats=1, completed_batches=5, completed_images=10, pending_batches=0),
        measurements=dict(value=2000., unit='images/s', status='measured', reason=None,
            elapsed_wall_seconds=.005, aggregation='total_completed_images_over_total_wall_seconds'),
        raw_evidence={'windows':[row]}, diagnostics=dict(warnings=[], limitations=[]))


def envelope(data=None):
    return dict(status='success', scenario=runner.P02_SCENARIO, executions=1,
                retries=0, baseline=0, throughput=payload() if data is None else data)


def make_root(root, scenario=runner.P02_SCENARIO):
    (root/'source').mkdir()
    sources = {}
    for name in ('gpu_scenario_runner.py','gpu_admission_policy.py','native_admission.py','p23_preflight.py'):
        path = root/'source'/name
        path.write_bytes((ROOT/'tools/forensics'/name).read_bytes())
        sources[name] = runner.digest(path)
    runtime = root/'runtime.bin'
    runtime.write_bytes(b'offline runtime identity fixture')
    manifest = dict(run_id='offline-only', limits=runner.LIMITS, scenario=scenario,
        executions=1, retries=0, baselines=0, admission_policy=runner.admission.POLICY_ID,
        source_files=sources, runtime_files={str(runtime):runner.digest(runtime)}, interpreter=sys.executable)
    if scenario == runner.P02_SCENARIO:
        manifest['p02_protocol'] = dict(runner.P02_PROTOCOL)
    runner.publish(root/'manifest.json', manifest)
    seal = dict(run_id=manifest['run_id'], manifest_sha256=runner.digest(root/'manifest.json'))
    runner.publish(root/'seal.json', seal)
    return manifest, seal


class P02RunnerExtensionTests(unittest.TestCase):
    def test_01_explicit_scenario_dispatch(self):
        tree = ast.parse(Path(runner.__file__).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='run_native_worker')
        dispatch = next(n for n in function.body if isinstance(n, ast.If) and 'P02_SCENARIO' in ast.unparse(n.test))
        code = compile(ast.Expression(dispatch.test), 'scenario_dispatch', 'eval')
        self.assertTrue(eval(code, dict(P02_SCENARIO=runner.P02_SCENARIO, manifest={'scenario':runner.P02_SCENARIO})))
        self.assertFalse(eval(code, dict(P02_SCENARIO=runner.P02_SCENARIO, manifest={'scenario':runner.SCENARIO})))
        self.assertIn('measure_p02_completion(model, inputs)', ast.unparse(dispatch))
        self.assertNotIn('BenchmarkRunner', ast.unparse(dispatch))

    def test_02_public_api_arguments_and_resident_tensor_construction(self):
        with public_schema() as (api, schema):
            api._reply = schema.ThroughputResult(payload())
            original = api.measure_throughput
            api.measure_throughput = Mock(wraps=original)
            model, inputs = object(), object()
            result = runner.measure_p02_completion(model, inputs)
            args, kwargs = api.measure_throughput.call_args
            inspect.signature(original).bind(*args, **kwargs)
            self.assertIs(args[0], model)
            self.assertIs(args[1], inputs)
            self.assertEqual(kwargs['protocol'], api.ThroughputProtocol(**runner.P02_PROTOCOL))
            self.assertEqual(result, payload())
            self.assertEqual(api.measure_throughput.call_count, 1)
        text = inspect.getsource(runner.run_native_worker)
        self.assertIn("torch.nn.Conv2d(1, 3, 3, padding=1).to('cuda:0').eval()", text)
        self.assertIn("torch.randn(2, 1, 7, 9, device='cuda:0', dtype=torch.float32)", text)
        self.assertIn('torch.manual_seed(71)', text)

    def test_03_production_result_serialization(self):
        with public_schema() as (_, schema):
            result = schema.ThroughputResult(payload())
            self.assertEqual(schema.ThroughputResult(json.loads(json.dumps(result.to_dict()))).to_dict(), payload())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'throughput.json'
                result.to_json(path)
                self.assertEqual(json.loads(path.read_text()), payload())

    def test_04_completed_pending_terminal_and_warmup_conservation(self):
        with public_schema() as (_, schema):
            schema.ThroughputResult(payload())
        runner.validate_result(envelope(), runner.P02_SCENARIO)
        data = payload()
        row = data['raw_evidence']['windows'][0]
        self.assertEqual(row['admitted_batches'], row['completed_batches']+row['pending_batches'])
        self.assertEqual(row['completed_images'], 2*row['completed_batches'])
        self.assertEqual(row['terminal_drains'], 1)
        self.assertEqual(data['measurements']['value'], row['completed_images']/row['elapsed_wall_seconds'])
        self.assertIn('warmup', data['protocol']['excluded'])
        self.assertEqual(data['requested']['warmup_iterations'], 1)

    def test_05_corrupt_completion_ledger_rejected(self):
        with public_schema() as (_, schema):
            for key, bad in [('pending_batches',1), ('completed_batches',4), ('completed_images',12),
                             ('terminal_drains',0), ('completion_groups',2), ('end_ns',1000000000)]:
                data = payload()
                data['raw_evidence']['windows'][0][key] = bad
                with self.subTest(key=key), self.assertRaises(ValueError):
                    schema.ThroughputResult(data)
                with self.subTest(transport=key), self.assertRaises(ValueError):
                    runner.validate_result(envelope(data), runner.P02_SCENARIO)

    def test_06_invalid_result_scope_rate_and_scenario_rejected(self):
        for key, bad in [('value',123), ('unit','ms'), ('status','derived'), ('reason','guessed')]:
            data = payload()
            data['measurements'][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_result(envelope(data), runner.P02_SCENARIO)
        with self.assertRaises(ValueError):
            runner.validate_result(envelope(), runner.SCENARIO)
        data = payload()
        data['provenance']['environment']['stream_id'] = 7
        with self.assertRaises(ValueError):
            runner.validate_p02_payload(data)

    def test_07_missing_execution_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_root(root)
            receipt = dict(outcome=None, exit_code=0)
            with patch.object(runner, 'supervise', return_value=receipt), patch('subprocess.Popen') as spawn:
                with self.assertRaises(FileNotFoundError):
                    runner.controller(root, root/'work', 'success')
                spawn.assert_not_called()
        with self.assertRaises(KeyError):
            runner.validate_gate({}, pid=1, parent_pid=2, limits=runner.expected_limits('child'), token='x')

    def test_08_source_runtime_and_manifest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, seal = make_root(root)
            runner.verify(root, seal['manifest_sha256'])
            for path in (root/'runtime.bin', root/'source/gpu_scenario_runner.py'):
                original = path.read_bytes()
                path.write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    runner.verify(root, seal['manifest_sha256'])
                path.write_bytes(original)
            with self.assertRaisesRegex(ValueError, 'manifest identity'):
                runner.verify(root, '0'*64)

    def test_09_isolation_invariants_preserved(self):
        old = ast.parse(Path(legacy.__file__).read_text())
        new = ast.parse(Path(runner.__file__).read_text())
        for name in ('job_class','expected_limits','validate_gate','await_gate','supervise'):
            a = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name==name)
            b = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(ast.parse(ast.unparse(a).replace('native_scenario_runner.py','gpu_scenario_runner.py'))),
                             ast.dump(ast.Module(body=[b], type_ignores=[])))
        self.assertEqual(runner.LIMITS, legacy.LIMITS)
        self.assertEqual((runner.LIMITS['child_timeout_s'],runner.LIMITS['outer_timeout_s']), (60,90))

    def test_10_previous_scenario_contract_still_accepted(self):
        result = dict(status='success', scenario=runner.SCENARIO, executions=1, baseline=0,
                      retries=0, benchmark={'preserved_legacy_payload':True})
        runner.validate_result(result, runner.SCENARIO)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, seal = make_root(root, runner.SCENARIO)
            self.assertNotIn('p02_protocol', manifest)
            runner.verify(root, seal['manifest_sha256'])

    @offline_test
    def test_11_no_execution_without_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_root(root)
            with patch('subprocess.Popen') as spawn:
                with self.assertRaisesRegex(ValueError, 'authorization required'):
                    runner.controller(root, root/'not-created', 'native')
                spawn.assert_not_called()
            self.assertFalse((root/'not-created').exists())
            self.assertFalse((root/'execution-consumed.json').exists())
        self.assertNotIn('torch', sys.modules)

    @offline_test
    def test_12_gpu_context_remains_warning_only(self):
        result = runner.admission.evaluate(None, {'gpu_processes':[{'pid':42,'category':'unknown'}]},
                    run_id='offline-only', manifest_sha256='a'*64)
        self.assertTrue(result['eligible'])
        self.assertIn('unknown_or_compute_gpu_process', result['warnings'])
        self.assertNotIn('torch', sys.modules)

    def test_13_p02_protocol_is_sealed_and_no_unplanned_scenario(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, seal = make_root(root)
            self.assertEqual(runner.verify(root, seal['manifest_sha256'])['p02_protocol'], runner.P02_PROTOCOL)
            for key, bad in [('queue_depth',4), ('repeats',2), ('warmup_iterations',True)]:
                changed = copy.deepcopy(manifest)
                changed['p02_protocol'][key] = bad
                (root/'manifest.json').write_text(json.dumps(changed))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'sealed protocol'):
                    runner.verify(root, runner.digest(root/'manifest.json'))
            manifest['scenario'] = 'unapproved-training'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'scenario/limit'):
                runner.verify(root, runner.digest(root/'manifest.json'))


if __name__ == '__main__':
    unittest.main()
