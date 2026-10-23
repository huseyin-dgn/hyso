"""Changed nested transport only: stdlib Windows Jobs, never Torch/CUDA."""
import ast
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
import uuid
from unittest.mock import patch

from test_p02_runner_extension import ROOT, make_root, public_schema
import gpu_scenario_runner as runner


@contextlib.contextmanager
def worker_schema():
    with public_schema() as (inference,result), patch.dict(sys.modules):
        inference.uuid=uuid
        tree=ast.parse((ROOT/'hyso/performance/inference.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_base')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_base','exec'),inference.__dict__)
        # Only import dependencies are stubs; real public protocols/results and
        # both worker launch dispatch paths are executed. No measured result.
        sys.modules['torch']=types.SimpleNamespace(__version__='offline-only')
        benchmark=types.ModuleType('hyso.benchmark.runner')
        benchmark._shape=lambda shape:tuple(shape)
        sys.modules[benchmark.__name__]=benchmark
        modules=[]
        for name in ('cold','capacity'):
            full='hyso.performance.'+name
            spec=importlib.util.spec_from_file_location(full,ROOT/'hyso/performance'/(name+'.py'))
            module=importlib.util.module_from_spec(spec);sys.modules[full]=module;spec.loader.exec_module(module);modules.append(module)
        yield *modules,result


class NestedWorkerTests(unittest.TestCase):
    def test_public_failure_schemas_and_scoped_dispatch(self):
        with worker_schema() as (cold,capacity,schema):
            seen=[]
            def launch(kind,recipe,plan,index,root):
                seen.append(kind)
                if kind=='capacity':
                    row=capacity._row(index,recipe)
                    row.update(status='crashed',reason='offline crash',exit_code=73,launcher_pid=123)
                    return row
                return dict(trial_index=index,status='crashed',worker_pid=None,launcher_pid=123,
                    exit_code=73,process_terminated=True,parent_start_ns=100,parent_end_ns=200,
                    parent_elapsed_ms=.0001,phases={n:cold._missing('crashed') for n in cold.PHASES},
                    startup_to_ready_receipt=dict(clock='parent_perf_counter_ns',status='not_observed',elapsed_ms=None))
            token=cold._owned_worker_launcher.set(launch)
            try:
                recipe=cold.ColdStartRecipe('torch.nn:Conv2d')
                p05=cold.measure_cold_start(recipe,protocol=cold.ColdStartProtocol(trials=1)).to_dict()
                p17=capacity.find_max_batch_size(recipe,protocol=capacity.BatchSearchProtocol(
                    candidates=[1,2],maximum_batch=4,cpu_memory_limit_bytes=4*1024**3,input_limit_bytes=1024,timeout_seconds=1)).to_dict()
                self.assertEqual(schema.ColdStartResult(p05).to_dict(),json.loads(json.dumps(p05)))
                self.assertEqual(schema.CapacityResult(p17).to_dict(),json.loads(json.dumps(p17)))
                self.assertEqual(seen,['cold','capacity'])
                self.assertEqual(p17['raw_evidence']['candidates'][1]['status'],'not_run')
                p05['actual']['measured']=1
                with self.assertRaises(ValueError):schema.ColdStartResult(p05)
                p17['measurements']['maximum_observed_feasible_batch']=2
                with self.assertRaises(ValueError):schema.CapacityResult(p17)
            finally:cold._owned_worker_launcher.reset(token)
            self.assertIsNone(cold._owned_worker_launcher.get())

    def run_transport(self,action,corrupt=None):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);manifest,seal=make_root(root)
            helper=root/'source/hyso/performance/_training_process.py';helper.parent.mkdir(parents=True)
            helper.write_bytes((ROOT/'hyso/performance/_training_process.py').read_bytes())
            manifest['source_files']['hyso/performance/_training_process.py']=runner.digest(helper)
            (root/'manifest.json').write_text(json.dumps(manifest));sha=runner.digest(root/'manifest.json')
            origin=root/'origin';origin.mkdir();work=root/'nested';work.mkdir()
            controller=dict(pid=111,manifest_sha256=sha,cuda_imported=False,source_verified=True,runtime_verified=True)
            parent=dict(pid=222,parent_pid=111,manifest_sha256=sha,cuda_imported=False,job_verified=True)
            worker=dict(pid=os.getpid(),parent_pid=222,manifest_sha256=sha,cuda_imported=False,job_verified=True,token='offline-token')
            if corrupt=='parent':worker['pid']+=1
            for name,data in [('controller',controller),('parent',parent),('worker',worker)]:runner.publish(origin/(name+'-identity.json'),data)
            runner.publish(work/'request.json',{})
            runner.publish(work/'nested-request.json',dict(kind='dryrun',origin_work=str(origin),action=action,
                payload_sha256='wrong' if corrupt=='payload' else runner.digest(work/'request.json')))
            if corrupt=='runtime':(root/'runtime.bin').write_bytes(b'mismatch')
            original_publish=runner.publish
            def publish(path,data):
                if corrupt=='creation' and Path(path).name=='worker-admit.json':
                    data['parent_identity']['creation_time_100ns']+=1
                return original_publish(path,data)
            with patch.object(runner,'publish',side_effect=publish):
                receipt=runner.supervise(root,work,'worker','success','offline-token',sha,timeout=2,nested=True)
            pre=runner.read(work/'worker-identity.json')
            self.assertFalse(pre['cuda_imported']);self.assertTrue(pre['job_verified'])
            self.assertEqual(receipt['cleanup']['active_processes'],0)
            self.assertTrue(receipt['owned_leader_exited']);self.assertTrue(receipt['kill_on_close'])
            self.assertEqual(receipt['assignment']['limits'],runner.expected_limits('child'))
            if corrupt:
                self.assertNotEqual(receipt['exit_code'],0)
                self.assertFalse((work/'nested-identity.json').exists())
            else:
                identity=runner.read(work/'nested-identity.json')
                self.assertFalse(identity['cuda_imported'])
                self.assertEqual(identity['parent'],runner.process_creation_identity())
                if action in ('success','oom','timeout'):
                    body=runner.read(work/'dryrun-result.json');self.assertFalse(body['cuda_imported'])
                    self.assertIn(body['descendant_pid'],receipt['owned_pids'])
                self.assertEqual(receipt['outcome'],'timeout' if action=='timeout' else None)
                self.assertEqual(receipt['exit_code'],73 if action=='crash' else 0)

    def test_preimport_admission_and_orphan_cleanup(self):self.run_transport('success')
    def test_timeout_and_tree_cleanup(self):self.run_transport('timeout')
    def test_crash_is_not_oom(self):self.run_transport('crash')
    def test_oom_transport_cleanup(self):self.run_transport('oom')
    def test_wrong_parent_rejected_before_import(self):self.run_transport('success','parent')
    def test_modified_request_rejected_before_import(self):self.run_transport('success','payload')
    def test_parent_creation_mismatch_rejected_before_import(self):self.run_transport('success','creation')
    def test_runtime_mismatch_rejected_before_import(self):self.run_transport('success','runtime')

    def test_new_scenarios_require_explicit_execution_receipt(self):
        for scenario in (runner.COLD_SCENARIO,runner.CAPACITY_SCENARIO):
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);manifest,_=make_root(root,scenario)
                manifest['scenario_contract']=runner.PREPARED[scenario]
                (root/'manifest.json').write_text(json.dumps(manifest))
                (root/'seal.json').write_text(json.dumps(dict(manifest_sha256=runner.digest(root/'manifest.json'))))
                with patch('subprocess.Popen') as launch:
                    with self.assertRaisesRegex(ValueError,'authorization required'):
                        runner.controller(root,root/'work','native')
                    launch.assert_not_called()
                self.assertFalse((root/'execution-consumed.json').exists())

    def test_missing_receipt_limit_and_pid_reuse_rejection(self):
        with self.assertRaises(KeyError):runner.validate_gate({},pid=1,parent_pid=2,limits={},token='x')
        row=dict(pid=1,parent_pid=2,limits=runner.expected_limits('child'),token='x',assigned=True,query_handle=1)
        row['limits']['process_bytes']=1
        with self.assertRaises(ValueError):runner.validate_gate(row,pid=1,parent_pid=2,limits=runner.expected_limits('child'),token='x')
        # Bootstrap must compare creation identity obtained from a live handle,
        # not re-open or terminate a process by an inventory PID.
        source=ast.unparse(ast.parse(Path(runner.__file__).read_text()))
        self.assertIn("process_creation_identity(receipt['parent_handle']) != receipt['parent_identity']",source)


if __name__=='__main__':unittest.main()
