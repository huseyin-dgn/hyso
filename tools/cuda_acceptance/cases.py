"""Real native cases, never imported by prepare-only or the CPU campaign runner."""
from pathlib import Path
import copy
import json
import threading
import time
import torch
from hyso import performance as perf
from hyso import quality
from hyso.benchmark import BenchmarkRunner
from . import workloads as w


class MissingEvidence(RuntimeError):pass


def require_cuda(data):
    device=data.get('provenance',{}).get('environment',{}).get('device')
    if device is None:device=data.get('environment',{}).get('device')
    if device!='cuda:0':raise AssertionError('native result is not CUDA: '+str(device))
    return data


def cohort():
    x,y=w.data()
    return quality.EvaluationCohort(x,y.tolist(),sample_ids=[str(i) for i in range(len(y))],
        class_names=['a','b'],split='holdout',preprocessing_id='fixed_arange_div_2048_v1')


def inference(ctx):
    from hyso import ModelBuilder,GraphModelBuilder
    def builder():
        m=ModelBuilder((1,16,16),2).add('conv2d',in_channels=1,out_channels=4,kernel_size=1,padding=0).build().to('cuda:0')
        x=w.data()[0][:2].to('cuda:0');out=m(x);out.sum().backward()
        assert out.device.type=='cuda' and out.shape==(2,2)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()) and torch.isfinite(out).all()
        return dict(device='cuda:0',shape=list(out.shape),gradients=True,finite=True)
    ctx.run('Builder',builder)
    def graph():
        b=GraphModelBuilder({'image':[1,16,16]}).add('conv','conv2d',['image'],in_channels=1,out_channels=4,kernel_size=1,padding=0)
        m=b.add('head','classification_head',['conv'],in_features=4,num_classes=2).outputs(logits='head').build().to('cuda:0')
        out=m({'image':w.data()[0][:2].to('cuda:0')})['logits'];out.sum().backward()
        assert out.device.type=='cuda' and out.shape==(2,2)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()) and torch.isfinite(out).all()
        return dict(device='cuda:0',shape=list(out.shape),gradients=True,finite=True)
    ctx.run('GraphBuilder',graph)
    def benchmark():
        master=ctx.shared('reference_model',w.model)
        resident=ctx.shared('resident_model',lambda:copy.deepcopy(master).to('cuda:0'))
        return BenchmarkRunner(resident,device='cuda:0',example_input=w.data()[0][:2].to('cuda:0'),
            warmup_iterations=2,measure_iterations=1000,repeats=1,measure_cuda_memory=True).run()
    def latency():
        result=perf.analyze_latency(ctx.shared('benchmark',benchmark),quantiles=(.5,.95,.99))
        d=require_cuda(result.to_dict());assert d['actual']['samples']==1000
        assert all(m['value'] is not None for m in d['measurements']);return d
    ctx.run('P01',latency)
    def throughput():
        r=perf.measure_throughput(w.model().to('cuda:0'),w.data()[0][:2].to('cuda:0'),
            protocol=perf.ThroughputProtocol(device='cuda:0',warmup_iterations=2,batch_count=1000,repeats=1))
        d=require_cuda(r.to_dict());assert d['actual']['completed_batches']==1000;return d
    ctx.run('P02',throughput)
    def vram():
        d=require_cuda(ctx.shared('benchmark',benchmark).to_dict())
        assert d['cuda_memory']['status']=='measured' and d['cuda_memory']['counter_reset_performed']
        assert d['cuda_memory']['peak_allocated_bytes']>0
        return d
    ctx.run('P09',vram,scope='owned_Torch_allocator_peak_not_physical_VRAM')
    ctx.run('P10',lambda:perf.measure_memory().to_dict(),scope='host_RSS_not_GPU_VRAM')
    def footprint():
        gpu=ctx.shared('benchmark',benchmark).to_dict();cpu=perf.measure_footprint(ctx.shared('reference_model',w.model)).to_dict()
        assert gpu['model']['weights_sha256']==cpu['provenance']['model']['weights_sha256']
        return dict(resident_GPU_source=gpu,logical_inventory=cpu)
    ctx.run('P11',footprint,scope='logical_inventory_bound_to_observed_GPU_model_not_VRAM')
    def pipeline():
        samples=iter(range(101))  # One warmup + 100 measured, unique IDs.
        def source():
            with torch.inference_mode(False),torch.no_grad():payload=w.data()[0][:1].clone()
            return perf.PipelineSample(str(next(samples)),payload)
        transfer=perf.PipelineStage('transfer',lambda s:perf.PipelineSample(s.sample_id,s.payload.to('cuda:0')),'cuda')
        return perf.measure_pipeline(w.model().to('cuda:0'),source,transfer=transfer,protocol=perf.PipelineProtocol(device='cuda:0',samples=100,warmup_samples=1))
    ctx.run('P03',lambda:require_cuda(ctx.shared('pipeline',pipeline).to_dict()),scope='pipeline_p50_p95_p99_eligibility_retained')
    ctx.run('P04',lambda:require_cuda(perf.analyze_stages(ctx.shared('pipeline',pipeline)).to_dict()))
    def cold():
        from .cold_probe import measure
        return measure(ctx)
    ctx.run('P05',cold,scope='one_fresh_owned_interpreter_no_OS_cache_coldness_claim')


def sweeps(ctx):
    def factory(shape,dtype,device):return torch.ones(shape,dtype=dtype,device=device)
    for feature,axis,points in [('P06','batch',[1,2,4]),('P07','resolution',[(16,16),(24,24)]),('P08','precision',['fp32','amp_fp16','bf16'])]:
        def capture(axis=axis,points=points):
            d=perf.run_sweep(w.model(),factory,axis=axis,points=points,input_shape=(2,1,16,16),
                protocol=perf.SweepProtocol(device='cuda:0',warmup_iterations=1,measure_iterations=20,repeats=1,continue_on_error=True)).to_dict()
            for p in d['raw_evidence']['points']:
                assert p['status']=='measured',p
                if p['result']['provenance']['environment']['device']!='cuda:0':raise AssertionError('sweep CPU fallback')
            return d
        ctx.run(feature,capture)
    def evaluation():
        from hyso.evaluation import evaluate_classification
        from torch.utils.data import DataLoader,TensorDataset
        x,y=w.data();r=evaluate_classification(w.model(),DataLoader(TensorDataset(x,y),batch_size=4),class_names=['a','b'],split='holdout',device='cuda:0')
        d=r.to_dict();assert d['source']['device']=='cuda:0';return d
    ctx.run('P26',evaluation)
    def precision():
        d=quality.evaluate_precision_quality(w.model(),cohort(),candidate='amp_fp16',device='cuda:0',batch_size=4).to_dict()
        assert d['sources']['candidate']['source']['device']=='cuda:0';return d
    ctx.run('P28',precision)


def profiling(ctx):
    recipe=perf.ColdStartRecipe('tools.cuda_acceptance.workloads:model',input_shape=(2,1,16,16),device='cuda')
    protocol=perf.ProfilingProtocol(measure_iterations=2,cpu_memory_limit_bytes=3*1024**3,input_limit_bytes=1024**2,cuda_memory_fraction=.15,timeout_seconds=90)
    def capture():return perf.profile_operators(recipe,protocol=protocol)
    def flops():
        d=require_cuda(perf.analyze_flops(ctx.shared('profile',capture)).to_dict())
        return d
    ctx.run('P12',flops,scope='observed_logical_ATen_cost_not_hardware_counters')
    def layers():
        d=require_cuda(perf.profile_layers(ctx.shared('profile',capture)).to_dict())
        if d['measurements'].get('cuda_event_timing',{}).get('status')!='measured':raise MissingEvidence('CUDA layer event timing unavailable')
        return d
    ctx.run('P13',layers,scope='inclusive_selected_stream_CUDA_events_not_exclusive_kernel_time')
    def operators():
        from .p14_probe import capture as capture_nsys
        return capture_nsys(ctx)
    ctx.run('P14',operators)
    def transfer():
        d=perf.measure_transfers(protocol=perf.TransferProtocol(shape=(65536,),pinned=True,non_blocking=True,
            warmup_iterations=1,measure_iterations=2,repeats=1,cpu_memory_limit_bytes=3*1024**3,input_limit_bytes=1024**2,cuda_memory_fraction=.15)).to_dict()
        return require_cuda(d)
    ctx.run('P15',transfer,scope='completed_copy_intervals_not_assumed_overlap')
    def utilization():
        m=w.model().to('cuda:0').eval();x=w.data()[0][:2].to('cuda:0')
        ready=threading.Event();stop=threading.Event();errors=[]
        activity=dict(pid=__import__('os').getpid(),device='cuda:0',uuid=ctx.uuid,
            completed=0,first_completion_ns=None,last_completion_ns=None,max_seconds=4.,exclusive=False)
        def work():
            try:
                deadline=time.monotonic()+4.
                with torch.inference_mode():
                    while True:
                        stopping=stop.is_set()
                        out=m(x);torch.cuda.synchronize()
                        assert torch.isfinite(out).all() and out.shape==(2,2)
                        stamp=time.perf_counter_ns();activity['completed']+=1
                        if activity['first_completion_ns'] is None:activity['first_completion_ns']=stamp
                        activity['last_completion_ns']=stamp;ready.set()
                        if stopping or time.monotonic()>=deadline:break
            except BaseException as exc:errors.append(repr(exc));ready.set()
        thread=threading.Thread(target=work,name='bounded-P16-workload');thread.start()
        try:
            if not ready.wait(5) or errors:raise RuntimeError('P16 workload startup: '+repr(errors))
            d=perf.measure_utilization(protocol=perf.UtilizationProtocol(duration_seconds=1.2,interval_seconds=.1,gpu=True,gpu_selector=ctx.uuid,workload_label='bounded_concurrent_inference')).to_dict()
        finally:
            stop.set();thread.join(timeout=5)
        if thread.is_alive() or errors:raise RuntimeError('P16 workload cleanup: '+repr(errors))
        if d['measurements']['gpu']['fields']['gpu_busy_percent']['status']!='measured':raise MissingEvidence('GPU utilization sensor samples unavailable/insufficient')
        return dict(sensors=d,concurrent_workload=activity,
            attribution='physical_board_samples_overlap_owned_workload;not_model_exclusive;unknown_sensor_internal_window')
    ctx.run('P16',utilization,scope='board_observation_not_model_exclusive')
    def capacity():
        indexed=perf.ColdStartRecipe('tools.cuda_acceptance.workloads:model',input_shape=(2,1,16,16),device='cuda:0')
        r=perf.find_max_batch_size(indexed,protocol=perf.BatchSearchProtocol(candidates=[1,2,4],maximum_batch=4,
            cpu_memory_limit_bytes=3*1024**3,input_limit_bytes=1024**2,cuda_memory_fraction=.15,timeout_seconds=60))
        d=require_cuda(r.to_dict());assert d['actual']['unknown']==0
        from .identity import gpu_uuid
        import os
        observed=gpu_uuid(torch.cuda.get_device_properties(0).uuid)
        assert torch.cuda.current_device()==0 and observed==ctx.uuid==ctx.runtime['uuid']==os.environ['CUDA_VISIBLE_DEVICES']
        assert all(row['forward_completed'] and row['provenance']['device']=='cuda:0' and
            row['sweep']['provenance']['environment']['device']=='cuda:0' for row in d['raw_evidence']['candidates'])
        return dict(result=d,device_binding=dict(index=0,observed_uuid=observed,visible_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
            admission_uuid=ctx.runtime['admission']['uuid']))
    ctx.run('P17',capacity,scope='finite_feasible_range_not_unbounded_hardware_maximum')


def training(ctx):
    def episode():
        t,l=w.training();return perf.measure_training_detailed(t,l,epochs=1)
    ctx.run('Trainer',lambda:ctx.shared('training',episode).timing.to_dict(),scope='ordinary_fresh_CUDA_training')
    for feature,field in [('P19','timing'),('P20','stages'),('P22','throughput')]:
        def capture(field=field):
            d=getattr(ctx.shared('training',episode),field).to_dict()
            assert d['provenance']['runtime']['device']=='cuda:0';return d
        ctx.run(feature,capture)
    def gradients():
        t,l=w.training(amp=True);r=perf.measure_gradient_diagnostics(t,l,epochs=1)
        d=r.to_dict()
        assert d['source_p19']['provenance']['runtime']['device']=='cuda:0'
        return d
    ctx.run('P25',gradients,scope='separate_fresh_AMP_episode')
    def memory(saved_activation_lifetime=False):
        assert saved_activation_lifetime is False,'BLOCKED_B4_UNRESOLVED'
        t,l=w.training();r=perf.measure_training_memory(t,l,epochs=1,protocol=perf.TrainingMemoryProtocol(device='cuda:0'))
        d=r.to_dict()
        assert d['source_p19']['provenance']['runtime']['device']=='cuda:0'
        assert d['activation']['status']=='unsupported' and d['protocol']['peak_reset_policy']=='never'
        return dict(saved_activation_lifetime=saved_activation_lifetime,B4='BLOCKED_B4_UNRESOLVED',basic=d)
    ctx.run('P23',memory,scope='basic_only_no_saved_tensor_hooks_no_B4_path')


def new_features(ctx):
    def loader():
        from torch.utils.data import DataLoader,TensorDataset
        t,_=w.training();x,y=w.data();l=DataLoader(TensorDataset(x,y),batch_size=4,pin_memory=True)
        d=perf.measure_training_with_loader(t,l,epochs=1).loader.to_dict()
        assert d['schema_version']==2 and d['h2d_transfer']['status']=='measured'
        assert len(d['h2d_transfer']['observations'])==4 and d['worker_service']['value'] is None
        return d
    ctx.run('P21',loader,scope='actual_main_wait_H2D_and_compute_not_invented_worker_service')
    def target():
        d=quality.measure_time_to_quality(w.time_target_factory,cohort(),seeds=[1,2],target=1.,epochs=1,device='cuda:0').to_dict()
        assert d['sources']['execution_device']=='cuda:0' and d['analysis']['censored']==2
        assert d['analysis']['reached_mean_seconds'] is None;return d
    ctx.run('P24',target)
    def robust():
        m=w.model();before={k:v.clone() for k,v in m.state_dict().items()}
        d=quality.evaluate_robustness(m,cohort(),w.perturb,corruption_id='fixed_noise',severity=.1,seed=8,device='cuda:0').to_dict()
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
        assert d['sources']['clean']['source']['device']==d['sources']['perturbed']['source']['device']=='cuda:0';return d
    ctx.run('P27',robust)
    def bundle():
        a=w.model();b=w.model();before=[{k:v.clone() for k,v in m.state_dict().items()} for m in (a,b)]
        from hyso.benchmark.runner import _profile,_descriptor
        def weight_hash(m):return _descriptor(*_profile(m,torch.float32,torch.device('cpu')))[1]
        hashes_before=[weight_hash(m) for m in (a,b)]
        d=quality.compare_performance_quality(a,b,cohort(),device='cuda:0',iterations=3,batch_size=4,
            memory_metric='cuda_allocated_peak_bytes',allow_cuda_peak_reset=True).to_dict()
        assert d['sources']['comparison_version']==4 and d['analysis']['comparison_version']==4
        assert d['analysis']['memory']['status']=='comparable' and d['analysis']['winner'] is None
        assert all(torch.equal(v,m.state_dict()[k]) for m,state in zip((a,b),before) for k,v in state.items())
        quality.QualityExperimentResult(d)  # Revalidate model/cohort/input/precision/timing/allocator links.
        for side in ('baseline','candidate'):
            proof=d['sources'][side]['gpu_provenance']
            from .identity import gpu_uuid
            assert gpu_uuid(proof['physical_gpu_uuid'])==ctx.uuid
            assert proof['cuda_build']==ctx.runtime['cuda_build'] and proof['torch_version']==ctx.runtime['torch_version']
        hashes_after=[weight_hash(m) for m in (a,b)]
        assert hashes_before==hashes_after
        return dict(comparison=d,source_weights_before=hashes_before,source_weights_after=hashes_after)
    ctx.run('P29',bundle)
    def video():
        import cv2
        import numpy as np
        path=ctx.folder/'owned.avi';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10.,(16,16))
        if not writer.isOpened():raise MissingEvidence('recorded-video encoder unavailable')
        try:
            for i in range(8):writer.write(np.full((16,16,3),i*20,dtype=np.uint8))
        finally:writer.release()
        recipe=perf.WorkloadRecipe('tools.cuda_acceptance.workloads:frame_processor',dict(gpu_uuid=ctx.uuid),gpu_uuid=ctx.uuid)
        d=perf.measure_streaming(perf.StreamSource('video_file',str(path.resolve()),'owned_8_frame_MJPG_fixture'),recipe,ctx.folder/'video-run',
            protocol=perf.StreamProtocol(max_frames=8,queue_capacity=2,process_memory_bytes=3*1024**3,job_memory_bytes=6*1024**3,max_processes=4)).to_dict()
        from common import new_json,file_hash
        new_json(ctx.folder/'P30-native-raw.json',d)
        a=d['analysis'];assert a['status']=='completed' and a['pending']==0 and a['processed']>0,dict(
            analysis=a,raw_evidence=str(ctx.folder/'P30-native-raw.json'),sha256=file_hash(ctx.folder/'P30-native-raw.json'))
        assert a['received']==a['processed']+a['dropped'] and d['evidence']['header']['processor']['device_id']==ctx.uuid
        return d
    ctx.run('P30',video)


def deployment(ctx):
    def trend():
        recipe=perf.WorkloadRecipe('tools.cuda_acceptance.workloads:stability_workload',dict(gpu_uuid=ctx.uuid),gpu_uuid=ctx.uuid)
        short=perf.measure_stability(recipe,ctx.folder/'trend',protocol=perf.StabilityProtocol(duration_seconds=3.,interval_seconds=.1,
            process_memory_bytes=3*1024**3,job_memory_bytes=6*1024**3,max_processes=4))
        d=perf.analyze_memory_trend(short).to_dict()
        assert d['evidence']['stability']['evidence']['header']['workload']['device_id']==ctx.uuid
        if any(d['analysis'][key]['status']!='descriptive_fit' for key in ('allocated','reserved')):
            raise MissingEvidence('P32 needs at least three real allocator samples in the work window')
        assert d['analysis']['allocated']['max_bytes']>d['analysis']['allocated']['min_bytes']
        return d
    ctx.run('P32',trend,scope='short_bounded_GPU_trend_not_7200s_or_causal_leak')
    def devices():
        recipe=perf.WorkloadRecipe('tools.cuda_acceptance.workloads:device_pair')
        d=perf.compare_devices(recipe,['cpu',ctx.uuid],ctx.folder/'devices',protocol=perf.DeviceProtocol(warmup=1,samples=5,
            cuda_memory_fraction=.15,process_memory_bytes=3*1024**3,job_memory_bytes=6*1024**3,max_processes=4)).to_dict()
        assert d['analysis']['comparable'] and all(x['status']=='measured' for x in d['analysis']['devices']);return d
    ctx.run('P34',devices)
    def providers():
        ort=ctx.plan['runtime']['ort_site']
        if not ort or not Path(ort).exists():raise MissingEvidence('explicit ORT CUDA runtime unavailable')
        identity_path=ctx.folder/'ort-loaded-runtime.json'
        recipe=perf.WorkloadRecipe('tools.cuda_acceptance.workloads:onnx_model',dict(identity_path=str(identity_path)),gpu_uuid=ctx.uuid)
        d=perf.profile_execution_providers(recipe,ctx.folder/'providers',providers=['CUDAExecutionProvider','CPUExecutionProvider'],
            expected={'*':'CUDAExecutionProvider'},optimization='disabled',runtime_search_path=ort,
            provider_options={'CUDAExecutionProvider':{'device_id':0,'gpu_mem_limit':256*1024**2}},
            protocol=perf.ProviderProtocol(warmup=1,runs=3,process_memory_bytes=3*1024**3,job_memory_bytes=6*1024**3,max_processes=4)).to_dict()
        from common import new_json,read_json,file_hash
        new_json(ctx.folder/'P35-native-raw.json',d)
        assert d['analysis']['status']=='expected_execution_observed' and any(e['provider']=='CUDAExecutionProvider' for e in d['analysis']['events']),dict(
            analysis=d['analysis'],raw_evidence=str(ctx.folder/'P35-native-raw.json'),sha256=file_hash(ctx.folder/'P35-native-raw.json'))
        from .native_identity import validate_ort_identity
        identity=read_json(identity_path)
        validate_ort_identity(identity,d,ctx.plan['feature_tools']['ort'])
        return dict(result=d,loaded_runtime=identity,planned_runtime=ctx.plan['feature_tools']['ort'])
    ctx.run('P35',providers)


GROUPS={'02-inference':inference,'03-sweeps':sweeps,'04-profiling':profiling,
    '05-training':training,'06-new-features':new_features,'07-deployment':deployment}
