"""Execute acceptance adapter routing with strict device doubles, no Torch/GPU."""
import contextlib,copy,sys,threading,types
from unittest.mock import patch
from test_infrastructure import Fixture
from full_acceptance import handlers,workloads


class InferenceDeviceTests(Fixture):
    def exercise(self,device):
        case=self;gpu=device!='cpu';calls=[];masters=[];forward=threading.Event()
        class Tensor:
            def __init__(self,shape=(10000,1,28,28),device='cpu',dtype='float32'):
                self.shape=tuple(shape);self.device=device;self.dtype=dtype
            def __len__(self):return self.shape[0]
            def __getitem__(self,key):
                start,stop,step=key.indices(len(self))
                return Tensor((len(range(start,stop,step)),*self.shape[1:]),self.device,self.dtype)
            def clone(self):return copy.deepcopy(self)
            def to(self,device=None,*,dtype=None):return Tensor(self.shape,device or self.device,dtype or self.dtype)
            def numpy(self):
                case.assertEqual(self.device,'cpu')
                return types.SimpleNamespace(tobytes=lambda:b'fixed-input')
            def repeat(self,repeats):return Tensor(tuple(a*b for a,b in zip(self.shape,repeats)),self.device,self.dtype)
            def argmax(self,axis):return Tensor((len(self),),self.device,'int64')
            def float(self):return Tensor(self.shape,self.device,'float32')
            def reshape(self,*shape):return Tensor(shape,self.device,self.dtype)
        class Model:
            def __init__(self,arm):self.arm=arm;self.device='cpu';self.dtype='float32';self.weights=[1,2,3]
            def to(self,device):self.device=device;return self
            def __call__(self,inputs):
                resident(self,inputs);calls.append(('utilization',self.arm));forward.set()
                return Tensor((len(inputs),10),self.device)
        def resident(model,inputs=None):
            case.assertEqual((model.device,model.dtype),(device,'float32'))
            if gpu:case.assertTrue(all(model is not master for master in masters))
            if inputs is not None:case.assertEqual((inputs.device,inputs.dtype),(device,'float32'))
        def master(model):case.assertEqual((model.device,model.dtype),('cpu','float32'))
        def protocol(**kwargs):return types.SimpleNamespace(**dict(dict(precision='fp32'),**kwargs))
        def load(m,arm):
            value=Model(arm);masters.append(value);return value
        class Runner:
            def __init__(self,model,**kw):
                resident(model,kw['example_input']);case.assertEqual(kw['example_input'].shape,(256,1,28,28))
                case.assertEqual(kw,dict(device=device,precision='fp32',example_input=kw['example_input'],
                    warmup_iterations=3,measure_iterations=1000,repeats=1,seed=20261030,measure_cuda_memory=gpu))
                calls.append(('latency',model.arm))
            def run(self):return object()
        def throughput(model,inputs,*,protocol):
            resident(model,inputs);case.assertEqual(inputs.shape,(256,1,28,28))
            case.assertEqual(vars(protocol),dict(device=device,precision='fp32',warmup_iterations=3,batch_count=40,repeats=1))
            calls.append(('throughput',model.arm));return object()
        def quality(model,loader,**kw):
            master(model);case.assertEqual(kw['device'],device)
            case.assertEqual(kw['capture_evidence'],not gpu)
            case.assertEqual(loader['batch_size'],256);case.assertFalse(loader['shuffle'])
            images,labels=loader['dataset'];case.assertEqual((len(images),len(labels)),(10000,10000))
            case.assertEqual((images.device,images.dtype,labels.device,labels.dtype),('cpu','float32','cpu','int64'))
            calls.append(('quality',model.arm));return types.SimpleNamespace(to_dict=lambda:dict(metrics=dict(accuracy=.98)))
        def precision_quality(model,cohort,**kw):
            master(model);case.assertEqual(kw['device'],device);case.assertEqual(kw['batch_size'],256)
            case.assertIn(kw['candidate'],('amp_fp16','bf16') if gpu else ('fp64','bf16'))
            calls.append(('precision-quality',model.arm,kw['candidate']));return object()
        def sweep(model,factory,*,axis,points,protocol,input_shape):
            master(model);case.assertEqual(vars(protocol),dict(device=device,precision='fp32',warmup_iterations=3,measure_iterations=20,repeats=1,sustained_batches=10))
            expected={'batch':[1,8,32],'resolution':[(28,28),(32,32),(28,36)],
                'precision':['fp32','amp_fp16','bf16'] if gpu else ['fp32','fp64','bf16']}
            case.assertEqual(points,expected[axis])
            for point in points:
                shape=list(input_shape)
                if axis=='batch':shape[0]=point
                if axis=='resolution':shape[2:]=point
                dtype='float64' if axis=='precision' and point=='fp64' else 'float32'
                inputs=factory(shape,dtype,device)
                case.assertEqual((inputs.shape,inputs.device,inputs.dtype),(tuple(shape),device,dtype))
            calls.append(('sweep',model.arm,axis));return object()
        def pipeline(model,source,*,transfer,postprocess,protocol):
            resident(model);case.assertEqual(vars(protocol),dict(device=device,precision='fp32',samples=40,repeats=1,warmup_samples=1))
            sample=source();case.assertEqual((sample.payload.device,sample.payload.dtype),('cpu','float32'))
            if gpu:
                case.assertEqual(transfer.device,'cuda');sample=transfer.function(sample)
            else:case.assertIsNone(transfer)
            resident(model,sample.payload)
            case.assertEqual(postprocess.device,'cuda' if gpu else 'cpu')
            output=postprocess.function(types.SimpleNamespace(sample_id=sample.sample_id,payload=Tensor((1,10),device)))
            case.assertEqual((output.payload.device,output.payload.dtype),(device,'float32'))
            calls.append(('pipeline',model.arm));return object()
        def utilization(**kwargs):
            case.assertTrue(forward.wait(1),'utilization driver did not call resident model double')
            return object()
        def transfers(*,protocol):
            case.assertEqual(protocol.shape,(256,1,28,28));case.assertEqual(protocol.warmup_iterations,3)
            case.assertEqual((protocol.measure_iterations,protocol.repeats),(20,1))
            calls.append(('transfer',protocol.pinned,protocol.non_blocking));return object()
        def module(name,**values):
            value=types.ModuleType(name);value.__dict__.update(values);return value
        protocols={name:protocol for name in ('ThroughputProtocol','MemoryProtocol','SweepProtocol','PipelineProtocol','UtilizationProtocol','TransferProtocol')}
        perf=module('hyso.performance',**protocols,analyze_latency=lambda *a,**k:object(),measure_throughput=throughput,
            measure_footprint=lambda model,**k:(master(model),object())[1],measure_memory=lambda *a,**k:object(),
            run_sweep=sweep,PipelineSample=lambda ident,payload:types.SimpleNamespace(sample_id=ident,payload=payload),
            PipelineStage=lambda name,function,device:types.SimpleNamespace(name=name,function=function,device=device),
            measure_pipeline=pipeline,analyze_stages=lambda *a:object(),measure_utilization=utilization,measure_transfers=transfers)
        fake_torch=module('torch',inference_mode=contextlib.nullcontext,get_num_threads=lambda:2,get_num_interop_threads=lambda:1,
            isfinite=lambda value:types.SimpleNamespace(all=lambda:types.SimpleNamespace(item=lambda:True)),
            cuda=types.SimpleNamespace(is_available=lambda:True,set_device=lambda index:None,synchronize=lambda index:None),
            nn=types.SimpleNamespace(functional=types.SimpleNamespace(interpolate=lambda images,*,size,**k:Tensor((*images.shape[:2],*size)))))
        replacements={'torch':fake_torch,'torch.utils':module('torch.utils'),
            'torch.utils.data':module('torch.utils.data',DataLoader=lambda dataset,**kw:dict(dataset=dataset,**kw),TensorDataset=lambda *values:values),
            'hyso':module('hyso',evaluate_classification=quality,EvaluationCohort=lambda *a,**k:object(),evaluate_precision_quality=precision_quality,
                evaluate_robustness=lambda model,*a,**k:(master(model),object())[1],compare_performance_quality=lambda *a,**k:object()),
            'hyso.benchmark':module('hyso.benchmark',BenchmarkRunner=Runner),'hyso.performance':perf,
            'hyso._evaluation_evidence':module('hyso._evaluation_evidence',model_hash=lambda model:(model.device,model.dtype,tuple(model.weights))),
            'hyso.performance._batch_worker':module('hyso.performance._batch_worker',_private_commit=lambda:123),
            'hyso.performance.devices':module('hyso.performance.devices',_cuda_hardware=lambda *a:dict(gpu_uuid='GPU-test'))}
        p=dict(batch=256,warmup=3,samples=1000,repeats=1,seed=20261030,completed_batches=40,quality_batch=256,
            robustness_samples=256,robustness_severity=.05,sweep_batches=[1,8,32],resolution_points=[[28,28],[32,32],[28,36]],
            precisions_cpu=['fp32','fp64','bf16'],precisions_cuda=['fp32','amp_fp16','bf16'])
        self.m.update(protocols=dict(inference=p),hardware=dict(gpu_uuid='GPU-test'),assets=dict(dataset=dict(preprocessing='fixed'),
            models={arm:dict(model_id=arm,path=arm,sha256='checkpoint-fixed') for arm in ('A','B')}))
        root=self.out/'inference';root.mkdir()
        with patch.dict(sys.modules,replacements),patch.object(workloads,'configure'),patch.object(workloads,'load_model',side_effect=load),\
             patch.object(workloads,'data',return_value=(Tensor(),Tensor((10000,),dtype='int64'))),\
             patch.object(handlers,'save',return_value={}),patch.object(handlers,'sha',return_value='checkpoint-fixed'):
            outcome=handlers.inference(self.m,root,device)
        self.assertEqual(outcome['status'],'PASS')
        for model in masters:
            self.assertEqual((model.device,model.dtype,model.weights),('cpu','float32',[1,2,3]))
        for arm in ('A','B'):
            for name in ('latency','throughput','quality','pipeline'):
                self.assertEqual(calls.count((name,arm)),1)
            self.assertIn(('utilization',arm),calls)
            for axis in ('batch','resolution','precision'):self.assertIn(('sweep',arm,axis),calls)
            for precision in p['precisions_cuda' if gpu else 'precisions_cpu'][1:]:
                self.assertIn(('precision-quality',arm,precision),calls)
        self.assertEqual([c for c in calls if c[0]=='transfer'],
            [('transfer',False,False),('transfer',True,False),('transfer',True,True)] if gpu else [])

    def test_all_cuda_submeasurements_respect_device_and_precision_contracts(self):self.exercise('cuda:0')
    def test_cpu_submeasurements_keep_full_protocol_and_master(self):self.exercise('cpu')
