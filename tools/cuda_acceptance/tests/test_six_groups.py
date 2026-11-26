"""Small CPU/offline regressions for the fourteen failed cases; never GPU evidence."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(REPO),str(REPO/'tools/final_acceptance')]
from campaign_preflight import runtime_plan
sys.path[:0]=[str(REPO),*runtime_plan()['runtime_paths']]
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import torch
if torch.version.cuda is not None:raise RuntimeError('Regressions require the existing CPU-only Torch runtime')
torch.set_num_threads(1)
from tools.cuda_acceptance import workloads as w,cases
from tools.cuda_acceptance.feature_evidence import verify_feature
from tools.cuda_acceptance.native_identity import validate_ort_identity
from tools.cuda_acceptance.six_group_review import unaffected_equivalent,EXISTING
from common import read_json,file_hash

GPU='GPU-9eec2f80-4c4e-a5cd-7091-03f0d013cf11'


class Context:
    def __init__(self,folder,selected):
        self.folder=Path(folder);self.selected=set(selected);self.cache={};self.results={};self.uuid=GPU
        self.runtime=dict(uuid=GPU,admission=dict(uuid=GPU))
        self.plan=dict(runtime=dict(ort_site='C:/Users/hdgn5/hyso-p35-runtime-env/Lib/site-packages'))
    def shared(self,key,fn):
        if key not in self.cache:self.cache[key]=fn()
        return self.cache[key]
    def run(self,name,fn,**kw):
        if name in self.selected:self.results[name]=fn()


class SixGroupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        guard=patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU initialization forbidden'))
        guard.start();self.addCleanup(guard.stop)

    def test_training_moves_all_tensors_before_optimizer(self):
        order=[];parameter=SimpleNamespace(device='cpu');buffer=SimpleNamespace(device='cpu')
        class Model:
            def to(self,device):
                order.append('move');parameter.device=buffer.device=device;return self
            def parameters(self):return [parameter]
        def optimizer(parameters,**kw):
            self.assertEqual(order,['move']);self.assertEqual(parameter.device,buffer.device)
            self.assertEqual(parameter.device,'cuda:0');order.append('optimizer')
            return SimpleNamespace(param_groups=[dict(params=list(parameters))])
        def trainer(model,optimizer,**kw):order.append('trainer');return SimpleNamespace(model=model,optimizer=optimizer)
        with patch.object(w,'model',return_value=Model()),patch.object(torch.optim,'SGD',side_effect=optimizer),patch('hyso.training.Trainer',side_effect=trainer):
            w.training(device='cuda:0')
        self.assertEqual(order,['move','optimizer','trainer'])

    def test_cpu_factory_matches_preserved_implementation(self):
        source=(REPO/'outputs/cuda-functional-acceptance-20261126/six-group-remediation/before-source/tools/cuda_acceptance/workloads.py').read_text(encoding='utf-8')
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='training')
        namespace=dict(vars(w));exec(compile(ast.Module(body=[node],type_ignores=[]),'preserved-training','exec'),namespace)
        for frozen in (False,True):
            old,_=namespace['training'](17,device='cpu',frozen=frozen);current,_=w.training(17,device='cpu',frozen=frozen)
            self.assertTrue(all(torch.equal(v,current.model.state_dict()[k]) for k,v in old.model.state_dict().items()))
            self.assertEqual({id(p) for p in current.model.parameters()},{id(p) for g in current.optimizer.param_groups for p in g['params']})

    def test_pipeline_source_and_shared_capture(self):
        ctx=Context(self.root,('P03','P04'));seen=[]
        result=SimpleNamespace(to_dict=lambda:dict(environment=dict(device='cuda:0')))
        def capture(model,source,transfer,protocol):
            seen.append('capture');ids=[]
            for _ in range(101):
                sample=source()
                self.assertIs(type(sample),cases.perf.PipelineSample);self.assertIs(type(sample.payload),torch.Tensor)
                self.assertEqual(sample.payload.device.type,'cpu');ids.append(sample.sample_id)
            self.assertEqual(len(set(ids)),101)
            with patch.object(torch.Tensor,'to',lambda tensor,*a,**kw:tensor):
                moved=transfer.function(sample)
            self.assertEqual(moved.sample_id,sample.sample_id);self.assertIs(moved.payload,sample.payload)
            self.assertEqual(protocol.device,'cuda:0');self.assertEqual(transfer.device,'cuda')
            return result
        with patch.object(torch.nn.Module,'to',lambda model,*a,**kw:model),patch.object(cases.perf,'measure_pipeline',side_effect=capture),patch.object(cases.perf,'analyze_stages',return_value=result):
            cases.inference(ctx)
        self.assertEqual(seen,['capture']);self.assertEqual(set(ctx.results),{'P03','P04'})

    def test_p17_indexed_recipe_and_physical_binding(self):
        ctx=Context(self.root,('P17',))
        rows=[dict(batch=n,status='feasible',forward_completed=True,provenance=dict(device='cuda:0'),
            sweep=dict(provenance=dict(environment=dict(device='cuda:0')))) for n in (1,2,4)]
        raw=dict(provenance=dict(environment=dict(device='cuda:0')),actual=dict(unknown=0),raw_evidence=dict(candidates=rows))
        def search(recipe,protocol):
            self.assertEqual(recipe.to_dict()['device'],'cuda:0');self.assertEqual(list(protocol.candidates),[1,2,4])
            return SimpleNamespace(to_dict=lambda:copy.deepcopy(raw))
        with patch.object(cases.perf,'find_max_batch_size',side_effect=search),patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(uuid=GPU[4:])),patch.object(torch.cuda,'current_device',return_value=0),patch.dict(os.environ,CUDA_VISIBLE_DEVICES=GPU):
            cases.profiling(ctx)
        verify_feature('P17',ctx.results['P17'],dict(uuid=GPU))
        bad=copy.deepcopy(ctx.results['P17']);bad['device_binding']['index']=1
        with self.assertRaises(ValueError):verify_feature('P17',bad,dict(uuid=GPU))
        for damage in ('unknown','incomplete'):
            bad=copy.deepcopy(ctx.results['P17'])
            if damage=='unknown':bad['result']['actual']['unknown']=1
            else:bad['result']['raw_evidence']['candidates'][0]['forward_completed']=False
            with self.assertRaises(ValueError):verify_feature('P17',bad,dict(uuid=GPU))
        with self.assertRaises(AssertionError):cases.require_cuda(dict(environment=dict(device='cuda')))

    def test_p14_closed_commands_have_bounded_capture_and_no_breakaway(self):
        from tools.cuda_acceptance.p14_probe import commands
        profile,export=commands('nsys.exe','python.exe','bootstrap.py','request.json',self.root)
        self.assertIn('--pytorch=autograd-nvtx',profile);self.assertIn('--capture-range=cudaProfilerApi',profile)
        self.assertIn('--capture-range-end=stop',profile);self.assertIn('--force-overwrite=false',profile)
        self.assertEqual(profile[-5:],['-I','-S','-B','bootstrap.py','request.json'])
        self.assertEqual(export[1:3],['export','--type=sqlite'])
        self.assertFalse(any('breakaway' in arg.lower() for arg in profile))

    def test_p14_replays_existing_real_kernel_chain_offline(self):
        from tools.cuda_acceptance.p14_probe import verify_capture
        historical=Path('C:/Users/hdgn5/hyso-private-evidence/p14-nsys-native-20261026')
        spec=importlib.util.spec_from_file_location('existing_p14_reader',REPO/'hyso/performance/_operator_activity.py')
        reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
        result=reader.analyze(historical/'p14.sqlite',historical/'capture.json')
        process=dict(pid=result['capture']['worker_pid']);runtime=dict(uuid=GPU,loaded_binaries={'pinned':'sha'},torch_version='2.5.1+cu121',cuda_build='12.1')
        receipt=dict(cleanup_verified=True,cleanup=dict(active_processes=0),target=process,limits=dict(process_bytes=3*1024**3,job_bytes=6*1024**3,max_processes=48))
        value=dict(result=result,target_runtime=dict(runtime,device_index=0,process=process,allocator_fraction=.15),
            profile_controller=receipt,export_controller=receipt,report=dict(path=str(historical/'p14.nsys-rep'),sha256=file_hash(historical/'p14.nsys-rep')))
        verify_capture(value,runtime)
        bad=copy.deepcopy(value);bad['result']['kernels']=[]
        with self.assertRaises(ValueError):verify_capture(bad,runtime)
        bad=copy.deepcopy(value);bad['profile_controller']['cleanup_verified']=False
        with self.assertRaises(ValueError):verify_capture(bad,runtime)

    def test_p30_uuid_is_canonical_but_wrong_device_stays_rejected(self):
        from hyso.performance.streaming import _physical_uuid,FrameInferenceProcessor
        for value in (GPU,GPU[4:]):self.assertEqual(_physical_uuid(value),GPU)
        for value in ('cuda:0','0',None):
            with self.assertRaises((ValueError,TypeError,AttributeError)):_physical_uuid(value)
        class AfterIdentity(Exception):pass
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=GPU),patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(uuid=GPU[4:])),patch.object(torch.cuda,'set_per_process_memory_fraction',side_effect=AfterIdentity):
            with self.assertRaises(AfterIdentity):FrameInferenceProcessor(torch.nn.Identity(),device='cuda:0',gpu_uuid=GPU,cuda_memory_fraction=.15)
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=GPU),patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(uuid='11111111-2222-4333-8444-555555555555')),patch.object(torch.cuda,'set_per_process_memory_fraction',side_effect=AssertionError('allocator before UUID')):
            with self.assertRaisesRegex(ValueError,'physical GPU UUID mismatch'):FrameInferenceProcessor(torch.nn.Identity(),device='cuda:0',gpu_uuid=GPU,cuda_memory_fraction=.15)

    def test_raw_results_survive_p30_and_p35_assertions(self):
        for feature,group,operation in [('P30',cases.new_features,'measure_streaming'),('P35',cases.deployment,'profile_execution_providers')]:
            with self.subTest(feature=feature):
                folder=self.root/feature;folder.mkdir();ctx=Context(folder,(feature,))
                raw=dict(analysis=dict(status='aborted',pending=0,processed=0,reason='synthetic_failure_before_processing'))
                with patch.object(cases.perf,operation,return_value=SimpleNamespace(to_dict=lambda:raw)):
                    with self.assertRaises(AssertionError) as error:group(ctx)
                path=folder/(feature+'-native-raw.json')
                self.assertEqual(read_json(path),raw);self.assertIn(file_hash(path),str(error.exception))

    def _provider_replay(self):
        from hyso.performance.runtime_result import _result
        base=REPO/'outputs/cuda-functional-acceptance-20261126/attempts/a506d330b0ee49e88b2c00df5c135e61/providers'
        request=read_json(base/'request.json')
        evidence={k:request[k] for k in ('token','recipe','protocol','providers','expected','optimization','runtime_search_path','provider_options')}
        evidence.update(controller=read_json(base/'controller.json'),report=read_json(base/'report.json'),graph_analysis_version=2,provider_analysis_version=2)
        return _result('P35',evidence).to_dict()

    def test_p35_real_saved_nodes_match_current_status_contract(self):
        result=self._provider_replay();a=result['analysis']
        self.assertEqual(a['status'],'expected_execution_observed');self.assertFalse(a['fallback_detected'])
        self.assertEqual(a['evidence_assessment']['node_coverage']['status'],'complete')
        self.assertEqual(len(a['events']),20);self.assertTrue(all(r['provider']=='CUDAExecutionProvider' for r in a['events']))
        # Identity fixtures only test the acceptance gate, never actual DLL proof.
        site='C:/pinned-ort';pid=result['evidence']['controller']['launcher_pid']
        binaries={str(Path(site)/name):'0'*64 for name in ('onnxruntime_providers_cuda.dll','onnxruntime_pybind11_state.pyd','nvcuda.dll','cudnn64_9.dll','cudart64_12.dll')}
        expected=dict(site=site,version=result['evidence']['report']['runtime']['version'],allowed_binaries=binaries)
        identity=dict(pid=pid,admitted_pid=str(pid),source=str(Path(site)/'onnxruntime/__init__.py'),version=expected['version'],visible_uuid=GPU,loaded_binaries=dict(binaries))
        value=dict(result=result,loaded_runtime=identity,planned_runtime=expected)
        verify_feature('P35',value,{})
        for damage in ('listed_only','fallback','DLL'):
            bad=copy.deepcopy(value)
            if damage=='listed_only':bad['result']['analysis']['events']=[]
            elif damage=='fallback':bad['result']['analysis']['events'][0]['provider']='CPUExecutionProvider'
            else:bad['loaded_runtime']['loaded_binaries'][next(iter(binaries))]='1'*64
            with self.assertRaises(ValueError):verify_feature('P35',bad,{})

    def test_accepted_scope_AST_paths_are_unchanged(self):
        for name in EXISTING:
            with self.subTest(name=name):
                before=(REPO/'outputs/cuda-functional-acceptance-20261126/six-group-remediation/before-source'/name).read_text(encoding='utf-8')
                after=(REPO/name).read_text(encoding='utf-8')
                self.assertTrue(unaffected_equivalent(before,after,Path(name).stem))
        before=(REPO/'tools/cuda_acceptance/cases.py').read_text(encoding='utf-8')
        self.assertFalse(unaffected_equivalent(before,before.replace("ctx.run('Builder',builder)","ctx.run('Builder',lambda:None)"),'cases'))

    def test_b4_and_trainer_product_guard_remain_closed(self):
        trainer=(REPO/'hyso/training/trainer.py').read_text(encoding='utf-8')
        self.assertIn('if tensor.device != self.device:',trainer)
        source=(REPO/'tools/cuda_acceptance/cases.py').read_text(encoding='utf-8')
        self.assertIn('assert saved_activation_lifetime is False',source)
        self.assertNotIn('saved_activation_lifetime=True',source)
        self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    result=unittest.main(exit=False)
    if torch.cuda.is_initialized():raise RuntimeError('Regression initialized CUDA')
    raise SystemExit(not result.result.wasSuccessful())
