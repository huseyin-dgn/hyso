"""Target only new prepared paths. Synthetic protocol data; no Torch/CUDA import."""
import ast
import copy
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, MagicMock, patch

from test_p02_runner_extension import public_schema, make_root, ROOT
import gpu_scenario_runner as runner
from offline_test_isolation import offline_test


@contextmanager
def pipeline_schema():
    with public_schema() as (inference, schema):
        module = types.ModuleType('hyso.performance.pipeline')
        source = ast.parse((ROOT/'hyso/performance/pipeline.py').read_text())
        tail = next(n for n in ast.parse((ROOT/'hyso/performance/inference.py').read_text()).body
                    if isinstance(n,ast.ClassDef) and n.name=='TailPolicy')
        names = {'PipelineSample','PipelineStage','PipelineProtocol','_callable',
                 '_metrics','_stage_metrics','_validate_pipeline','analyze_stages'}
        nodes = [tail]+[n for n in source.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
        module.__dict__.update(dataclass=dataclass, asdict=asdict, math=math, inspect=inspect,
            copy=copy, torch=types.SimpleNamespace(Tensor=object),
            ThroughputProtocol=inference.ThroughputProtocol, _integer=schema._integer,
            _latency_metrics=schema._latency_metrics, PipelineResult=schema.PipelineResult,
            StageTimingResult=schema.StageTimingResult)
        with patch.dict(sys.modules, {'hyso.performance.pipeline':module}):
            exec(compile(ast.Module(body=nodes,type_ignores=[]),'production_pipeline_schema','exec'),module.__dict__)
            yield module, schema


def pipeline_payload(module, schema):
    rows=[]
    names=['source','preprocess','transfer','inference','postprocess','deliver']
    for index in range(1,4):
        base=index*100000000
        stages=[]
        for i,name in enumerate(names):
            start=base+i*1000000+(100 if name=='deliver' else 0)
            cuda=name in ('transfer','inference','postprocess')
            stages.append(dict(name=name,sample_id='frame-'+str(index),start_ns=start,
                launch_end_ns=start+500000,completion_ns=start+1000000,host_launch_ms=.5,host_elapsed_ms=1.,
                device_event_ms=.25 if cuda else None,device_event_status='measured' if cuda else 'not_applicable',
                device_event_reason=None if cuda else 'synchronous_host_stage',device='cuda' if cuda else 'cpu',
                stream_id=0 if cuda else None,sync_policy='end_event_sync' if cuda else 'synchronous_callable',
                status='measured',reason=None,unit='ms'))
        rows.append(dict(sample_id='frame-'+str(index),repeat_index=0,source_start_ns=base,
            source_completion_ns=base+1000000,postprocess_completion_ns=base+5000100,
            delivery_completion_ns=base+6000100,queue_wait_ns=0,e2e_ms=5.0001,
            status='completed',reason=None,unit='ms',stages=stages))
    data=dict(format='hyso.performance.pipeline',schema_version=1,result_id='11111111-1111-4111-8111-111111111111',
        feature_ids=['P03'],identity=dict(experiment_run_id=None,source_result_id=None),
        protocol=dict(id='serial_bounded_pipeline',version=1,scope='read_start_to_post_completion',
            configuration=runner.PREPARED[runner.PIPELINE_SCENARIO]['protocol'],clock='perf_counter_ns',
            queue_policy='block',stage_order=names,stage_devices=['cpu','cpu','cuda','cuda','cuda','cpu']),
        provenance=dict(model={'fixture':'synthetic'},input={'fixture':'synthetic'},environment=dict(device='cuda:0',stream_id=0)),
        requested=dict(samples=3,repeats=1,warmup_samples=1),
        actual=dict(completed=3,dropped=0,pending=0,warmup_samples=1,forward_count=4,max_pending=2),
        measurements=module._metrics(rows),raw_evidence=dict(samples=rows),diagnostics=dict(warnings=[],limitations=[]))
    p=schema.PipelineResult(data)
    return {'P03':p.to_dict(),'P04':module.analyze_stages(p).to_dict()}


def frames():
    ids=['frame-'+str(i) for i in range(4)]
    return dict(source_ids=ids,delivered_ids=ids)


class Box:
    def __init__(self,data):self.data=copy.deepcopy(data)
    def to_dict(self):return copy.deepcopy(self.data)


class PreparedValidationTests(unittest.TestCase):
    def test_pipeline_real_schema_and_shared_stage_serialization(self):
        with pipeline_schema() as (module,schema):
            data=pipeline_payload(module,schema)
            schema.PipelineResult(json.loads(json.dumps(data['P03'])))
            schema.StageTimingResult(json.loads(json.dumps(data['P04'])))
            runner.validate_pipeline_pair(data,frames())
            self.assertIsNone(data['P03']['measurements']['p99_ms']['value'])
            self.assertEqual(data['P04']['identity']['source_result_id'],data['P03']['result_id'])

    def test_pipeline_public_api_once_and_all_stage_identities(self):
        with pipeline_schema() as (module,schema):
            data=pipeline_payload(module,schema)
            class Tensor:
                def __truediv__(self,value):return self
                def to(self,*a,**k):return self
                def mean(self,*a,**k):return self
                def cpu(self):return self
            def api(model,source,*,preprocess,transfer,postprocess,deliver,protocol):
                self.assertEqual(asdict(protocol),runner.PREPARED[runner.PIPELINE_SCENARIO]['protocol'])
                for _ in range(4):
                    frame=source()
                    for stage in (preprocess,transfer,postprocess,deliver):frame=stage.function(frame)
                return schema.PipelineResult(data['P03'])
            module.measure_pipeline=Mock(side_effect=api)
            torch=MagicMock();torch.full.return_value=Tensor()
            raw,ids=runner.measure_pipeline_pair(torch,runner.PREPARED[runner.PIPELINE_SCENARIO])
            self.assertEqual(module.measure_pipeline.call_count,1)
            self.assertEqual(ids,frames())
            runner.validate_pipeline_pair(raw,ids)

    def test_corrupt_frame_clock_and_cuda_evidence_rejected(self):
        with pipeline_schema() as (module,schema):
            source=pipeline_payload(module,schema)
            for change in ('frame','host','cuda','association','summary'):
                raw=copy.deepcopy(source)
                if change=='frame':raw['P03']['raw_evidence']['samples'][0]['sample_id']='other'
                elif change=='host':raw['P03']['raw_evidence']['samples'][0]['stages'][1]['host_elapsed_ms']=999
                elif change=='cuda':raw['P03']['raw_evidence']['samples'][0]['stages'][2]['device_event_ms']=None
                elif change=='association':raw['P04']['identity']['source_result_id']='other'
                else:raw['P04']['measurements']['preprocess']['host_mean_ms']=999
                with self.subTest(change=change),self.assertRaises(ValueError):runner.validate_pipeline_pair(raw,frames())

    def test_dispatch_uses_public_payload_and_digest(self):
        with pipeline_schema() as (module,schema):
            raw=pipeline_payload(module,schema)
            with patch.object(runner,'measure_pipeline_pair',return_value=(raw,frames())) as call:
                result=runner.run_prepared_scenario(None,{'scenario':runner.PIPELINE_SCENARIO},{'warnings':[],'measurement_reliability':{'status':'observed'}})
            self.assertEqual(call.call_count,1)
            self.assertNotIn('benchmark',result)
            runner.validate_result(result,runner.PIPELINE_SCENARIO)
            result['public_result_sha256']='tampered'
            with self.assertRaises(ValueError):runner.validate_result(result,runner.PIPELINE_SCENARIO)

    def test_sweep_arguments_for_each_real_public_axis(self):
        # Only API wiring is spied; no claim of native sweep validation.
        with public_schema() as (inference,schema):
            module=types.ModuleType('hyso.performance.sweeps')
            tree=ast.parse((ROOT/'hyso/performance/sweeps.py').read_text())
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SweepProtocol')
            module.__dict__.update(dataclass=dataclass,ThroughputProtocol=inference.ThroughputProtocol,
                _integer=schema._integer,_PRECISIONS=('fp32','amp_fp16'))
            exec(compile(ast.Module(body=[cls],type_ignores=[]),'sweep_protocol','exec'),module.__dict__)
            with patch.dict(sys.modules,{'hyso.performance.sweeps':module}),patch.object(schema,'SweepResult',side_effect=Box):
                for scenario,contract in runner.PREPARED.items():
                    if contract['kind']!='sweep':continue
                    reply=dict(feature_ids=[contract['feature']])
                    def api(model,factory,*,axis,points,input_shape,protocol):
                        self.assertEqual(axis,contract['axis']);self.assertEqual(points,contract['points'])
                        self.assertEqual(asdict(protocol),contract['protocol'])
                        factory(input_shape,'dtype','device')
                        return Box(reply)
                    module.run_sweep=Mock(side_effect=api)
                    torch=MagicMock()
                    raw,_=runner.measure_sweep_case(torch,contract)
                    self.assertEqual(module.run_sweep.call_count,1)
                    self.assertEqual(raw[contract['feature']],reply)
                    torch.randn.assert_called_once_with((2,1,7,9),dtype='dtype',device='device')

    def test_training_bundle_has_one_public_fit_and_source_projections(self):
        timing={'segment_id':'one-fit'}
        data={'source_p19':timing,'source_p20':{'source_segment_id':'one-fit'}}
        gradients=types.ModuleType('hyso.performance.gradients')
        timing_module=types.ModuleType('hyso.performance.training')
        details=types.ModuleType('hyso.performance.training_details')
        training=types.ModuleType('hyso.training')
        gradients.measure_gradient_diagnostics=Mock(return_value=Box(data))
        gradients.GradientDiagnosticsProtocol=lambda **k:k
        gradients.GradientDiagnosticsResult=Box
        timing_module.TrainingTimingProtocol=lambda **k:k
        timing_module.TrainingPerformanceResult=Box
        details.TrainingStageResult=details.TrainingThroughputResult=Box
        details.analyze_training_throughput=Mock(return_value=Box({'source_segment_id':'one-fit'}))
        training.Trainer=Mock()
        torch=MagicMock()
        torch.linspace.return_value.reshape.return_value.split.return_value=[1,2,3]
        torch.tensor.return_value.split.return_value=[0,1,0]
        modules={'hyso':types.ModuleType('hyso'),'hyso.performance':types.ModuleType('hyso.performance'),
            'hyso.training':training,'hyso.performance.gradients':gradients,
            'hyso.performance.training':timing_module,'hyso.performance.training_details':details}
        with patch.dict(sys.modules,modules), patch.object(runner,'training_state_snapshot',
                side_effect=[{'sha256':'before'},{'sha256':'after'}]) as snapshot:
            raw,evidence=runner.measure_training_bundle(torch,runner.PREPARED[runner.TRAINING_SCENARIO])
        self.assertEqual(set(raw),{'P19','P20','P22','P25'})
        self.assertEqual(gradients.measure_gradient_diagnostics.call_count,1)
        self.assertEqual(details.analyze_training_throughput.call_count,1)
        self.assertEqual(evidence['fits'],1)
        self.assertEqual(snapshot.call_count,2)
        self.assertEqual(evidence['before']['sha256'],'before')
        self.assertEqual(evidence['after']['sha256'],'after')
        self.assertFalse(evidence['configuration']['amp'])
        self.assertFalse(training.Trainer.return_value.fit.called)

    def test_training_state_snapshot_serialization_and_limits(self):
        class Tensor:
            dtype='float32';device='cuda:0';shape=(2,)
            def numel(self):return 2
            def detach(self):return self
            def cpu(self):return self
            def tolist(self):return [1.,2.]
        tensor=Tensor()
        torch=types.SimpleNamespace(is_tensor=lambda v:isinstance(v,Tensor),
            get_rng_state=lambda:tensor,cuda=types.SimpleNamespace(get_rng_state=lambda d:tensor))
        trainer=types.SimpleNamespace(model=types.SimpleNamespace(state_dict=lambda:{'weight':tensor}),
            optimizer=types.SimpleNamespace(state_dict=lambda:{'state':{},'param_groups':[{'lr':.01}]}))
        with patch.dict(sys.modules,{'numpy':None}):
            before=runner.training_state_snapshot(torch,trainer)
            self.assertEqual(json.loads(json.dumps(before)),before)
            self.assertEqual(before['state']['model']['weight']['values'],[1.,2.])
            tensor.numel=lambda:16385
            with self.assertRaisesRegex(ValueError,'tensor budget'):
                runner.training_state_snapshot(torch,trainer)

    def test_prepared_contracts_sealed_and_blocked_child_worker_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,_=make_root(root)
            for scenario,contract in runner.PREPARED.items():
                manifest.update(scenario=scenario,scenario_contract=contract)
                (root/'manifest.json').write_text(json.dumps(manifest))
                runner.verify(root,runner.digest(root/'manifest.json'))
                altered=copy.deepcopy(manifest);altered['scenario_contract']['seed']=99
                (root/'manifest.json').write_text(json.dumps(altered))
                with self.subTest(scenario=scenario),self.assertRaises(ValueError):
                    runner.verify(root,runner.digest(root/'manifest.json'))
        self.assertEqual(runner.BLOCKED,{})
        self.assertIn(runner.COLD_SCENARIO,runner.SCENARIOS)
        self.assertIn(runner.CAPACITY_SCENARIO,runner.SCENARIOS)

    @offline_test
    def test_pipeline_approval_absent_or_wrong_scenario_cannot_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest,seal=make_root(root)
            manifest.update(scenario=runner.PIPELINE_SCENARIO,scenario_contract=runner.PREPARED[runner.PIPELINE_SCENARIO])
            (root/'manifest.json').write_text(json.dumps(manifest));seal['manifest_sha256']=runner.digest(root/'manifest.json')
            (root/'seal.json').write_text(json.dumps(seal))
            with patch('subprocess.Popen') as launch:
                with self.assertRaisesRegex(ValueError,'authorization required'):
                    runner.controller(root,root/'work','native')
                approval=root/'approval.json'
                approval.write_text(json.dumps(dict(run_id=manifest['run_id'],manifest_sha256=seal['manifest_sha256'],
                    scenario=runner.P02_SCENARIO,maximum_executions=1,explicit_user_approval=True)))
                with self.assertRaisesRegex(ValueError,'approval identity'):
                    runner.controller(root,root/'work','native',approval)
                launch.assert_not_called()
            self.assertFalse((root/'execution-consumed.json').exists())
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
