"""Exercise the runner/API boundary without native workloads in unit tests."""
import sys,types
from unittest.mock import patch
from test_infrastructure import Fixture
from full_acceptance import handlers,cli,safety

class BenchmarkBoundaryTests(Fixture):
    def test_short_check_requires_explicit_hash(self):
        args=types.SimpleNamespace(manifest=self.path,approved_sha256=None)
        with patch.object(cli,'verify') as verify,patch.object(safety,'supervise') as launch:
            with self.assertRaisesRegex(ValueError,'requires exact'):cli.smoke(args,benchmark_check=True)
        verify.assert_not_called();launch.assert_not_called()

    def test_real_input_is_exclusive_and_protocol_is_preserved(self):
        sample=object();model=object();calls=[];answer=object()
        class Runner:
            def __init__(self,actual,**kw):
                if (kw.get('input_shape') is None)==(kw.get('example_input') is None):
                    raise ValueError('provide exactly one of input_shape or example_input')
                calls.append((actual,kw))
            def run(self):return answer
        module=types.ModuleType('hyso.benchmark');module.BenchmarkRunner=Runner
        with patch.dict(sys.modules,{'hyso.benchmark':module}):
            for device in ('cpu',):
                p=dict(warmup=3,samples=1000,repeats=1,seed=20261030)
                self.assertIs(handlers.inference_benchmark(model,sample,p,device),answer)
                actual,kw=calls[-1];self.assertIs(actual,model);self.assertIs(kw['example_input'],sample)
                self.assertNotIn('input_shape',kw)
                self.assertEqual(kw,dict(device=device,precision='fp32',example_input=sample,
                    warmup_iterations=3,measure_iterations=1000,repeats=1,seed=20261030,measure_cuda_memory=device!='cpu'))

    def test_cuda_benchmark_owns_resident_copy_and_preserves_cpu_master(self):
        class Model:
            def __init__(self):self.device='cpu';self.weights=[1,2,3]
            def to(self,device):self.device=device;return self
        class Tensor:
            def __init__(self,device='cpu'):self.device=device;self.values=[4,5,6]
            def to(self,device):return Tensor(device)
        model=Model();sample=Tensor();calls=[]
        class Runner:
            def __init__(self,actual,**kw):
                if actual.device!=kw['device'] or kw['example_input'].device!=kw['device']:
                    raise ValueError('benchmark requires dense initialized ordinary tensors on selected device')
                calls.append((actual,kw))
            def run(self):return 'measured'
        module=types.ModuleType('hyso.benchmark');module.BenchmarkRunner=Runner
        protocol=dict(warmup=3,samples=1000,repeats=1,seed=20261030)
        with patch.dict(sys.modules,{'hyso.benchmark':module}):
            self.assertEqual(handlers.inference_benchmark(model,sample,protocol,'cuda:0'),'measured')
        resident,kw=calls[0]
        self.assertIsNot(resident,model);self.assertIsNot(kw['example_input'],sample)
        self.assertEqual((model.device,sample.device),('cpu','cpu'))
        self.assertEqual(resident.weights,model.weights);self.assertEqual(kw['example_input'].values,sample.values)
        self.assertEqual((kw['precision'],kw['warmup_iterations'],kw['measure_iterations'],kw['repeats'],kw['seed'],kw['measure_cuda_memory']),
            ('fp32',3,1000,1,20261030,True))

    def test_short_check_rejects_failed_verification_before_supervision(self):
        args=types.SimpleNamespace(manifest=self.path,approved_sha256=self.hash)
        with patch.object(cli,'verify',side_effect=ValueError('hash rejected')),patch.object(safety,'supervise') as launch:
            with self.assertRaisesRegex(ValueError,'hash rejected'):cli.smoke(args,benchmark_check=True)
        launch.assert_not_called()

    def test_short_check_cannot_write_started_run(self):
        (self.out/'STARTED.json').write_text('{}')
        args=types.SimpleNamespace(manifest=self.path,approved_sha256=self.hash)
        with patch.object(cli,'verify',return_value=self.m),patch.object(safety,'supervise') as launch:
            with self.assertRaises(FileExistsError):cli.smoke(args,benchmark_check=True)
        launch.assert_not_called()

    def test_short_check_does_not_dispatch_acceptance_suite(self):
        args=types.SimpleNamespace(manifest=self.path,approved_sha256=self.hash)
        with patch.object(cli,'verify',return_value=self.m),patch.object(safety,'supervise',return_value={'status':'PASS'}) as launch,\
             patch.object(cli,'read',return_value=dict(experiments=[],preparation={})),\
             patch.object(cli,'write_reports'),patch.object(cli,'publish'),patch.object(cli,'preserve'):
            self.assertEqual(cli.smoke(args,benchmark_check=True),0)
        launch.assert_called_once()
        self.assertEqual(launch.call_args.kwargs['operation'],'cpu_benchmark_check')
        self.assertEqual(launch.call_args.args[1]['id'],'__cpu_benchmark_check')
        self.assertFalse((self.out/'STARTED.json').exists())

    def test_cuda_check_requires_hash_approval_and_fresh_unstopped_run(self):
        args=types.SimpleNamespace(manifest=self.path,approved_sha256=None)
        with patch.object(cli,'verify',return_value=self.m),patch.object(safety,'supervise') as launch:
            with self.assertRaisesRegex(ValueError,'requires exact'):cli.cuda_check(args)
            args.approved_sha256=self.hash
            with self.assertRaisesRegex(PermissionError,'NOT_AUTHORIZED'):cli.cuda_check(args)
            next(e for e in self.m['experiments'] if e['id']=='04.cuda')['authorization']['approved']=True
            (self.out/'STARTED.json').write_text('{}')
            with self.assertRaises(FileExistsError):cli.cuda_check(args)
            (self.out/'STARTED.json').unlink()
            (self.out/'stop-request.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'stop request'):cli.cuda_check(args)
        launch.assert_not_called()

    def test_cuda_check_uses_owned_supervisor_and_never_executes_suite(self):
        args=types.SimpleNamespace(manifest=self.path,approved_sha256=self.hash)
        next(e for e in self.m['experiments'] if e['id']=='04.cuda')['authorization']['approved']=True
        with patch.object(cli,'verify',return_value=self.m),patch.object(safety,'supervise',return_value={'status':'PASS'}) as launch,\
             patch.object(cli,'publish'),patch.object(cli,'preserve'),patch.object(cli,'execute') as execute:
            self.assertEqual(cli.cuda_check(args),0)
        execute.assert_not_called();launch.assert_called_once()
        self.assertEqual(launch.call_args.kwargs['operation'],'cuda_benchmark_check')
        self.assertEqual(launch.call_args.kwargs['stop_file'],self.out/'stop-request.json')
        self.assertEqual(launch.call_args.args[1]['outer_seconds'],180)
        self.assertFalse((self.out/'STARTED.json').exists())
