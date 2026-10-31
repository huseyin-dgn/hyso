"""Concrete adapters to the existing HYSo APIs. No optimizer/advisor engine."""
from pathlib import Path
import contextlib,csv,hashlib,importlib,importlib.metadata,json,math,os,statistics,subprocess,sys,tempfile,threading,time,uuid
from .core import read,sha,publish,record,now

def metric(name,value,unit,kind,scope,reason=None):
    if value is None:
        if not reason:raise ValueError('missing metric requires reason: '+name)
        kind='missing'
    return dict(name=name,value=value,unit=unit,kind=kind,scope=scope,reason=reason)
def result(status='PASS',*,reason=None,metrics=(),criteria=None,**extra):
    return dict(status=status,reason=reason,metrics=list(metrics),criteria=criteria or {},**extra)
def save(root,name,value):
    path=Path(root)/(name+'.json')
    if hasattr(value,'to_json'):
        value.to_json(path)
        if hasattr(value,'to_csv'):value.to_csv(path.with_suffix('.csv'))
    else:publish(path,value,new=True)
    return record(path)

def environment(m,root):
    import torch,torchvision,numpy
    from .workloads import configure
    configure(m)
    identity=dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),torchvision=str(torchvision.__version__),
        numpy=numpy.__version__,torch_file=torch.__file__,cuda_initialized=torch.cuda.is_initialized(),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),platform=sys.platform,logical_processors=os.cpu_count())
    if identity['torch']!=m['runtime']['expected']['torch'] or identity['torchvision']!=m['runtime']['expected']['torchvision']:
        raise ValueError('scientific runtime version mismatch')
    if torch.cuda.is_initialized():raise ValueError('metadata preflight initialized CUDA')
    if Path(torch.__file__).resolve().parent.parent!=Path(m['runtime']['site_paths'][0]).resolve():raise ValueError('wrong primary Torch runtime')
    with numpy.load(m['assets']['split_indices']['path'],allow_pickle=False) as data:
        split_keys=list(data.files)
        if 'train' not in data.files or not ('val' in data.files or 'validation' in data.files):raise ValueError('training split keys unavailable')
    identity['training_split_keys']=split_keys
    ref=save(root,'environment',identity)
    return result(evidence=[ref],criteria=dict(environment_identity=True,CUDANativeStarted=False),
        metrics=[metric('CUDA_initialized',False,'boolean','measured','current CPU metadata process')])

def smoke_cpu(m,root):
    import torch
    from .workloads import configure,load_model,data
    from hyso._evaluation_evidence import model_hash
    configure(m);before=torch.get_rng_state().clone()
    model=load_model(m,'B');x,_=data(m,count=1);state=model_hash(model)
    begin=time.perf_counter_ns()
    with torch.inference_mode():y=model(x)
    elapsed=time.perf_counter_ns()-begin
    if list(y.shape)!=[1,10] or not torch.isfinite(y).all().item():raise ValueError('trained CPU output contract')
    if model_hash(model)!=state or torch.cuda.is_initialized():raise ValueError('smoke changed model or initialized CUDA')
    return result(criteria=dict(trained_checkpoint_loaded=True,one_CPU_forward=True,output_contract=True,source_state_preserved=True),
        model='B',batch=1,device='cpu',acceptance_scope='one-forward infrastructure smoke only; not full inference acceptance',
        identity=dict(checkpoint=m['assets']['models']['B'],loaded_state_sha256=state,input_shape=list(x.shape),input_dtype=str(x.dtype),
            input_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),preprocessing=m['assets']['dataset']['preprocessing']),
        metrics=[metric('completed_forwards',1,'call','measured','smoke'),metric('host_forward_ms',elapsed/1e6,'ms','measured','one smoke call; no benchmark quantile')])

def regression(m,root):
    import pytest
    from .registry import SAFE_TESTS
    temp=Path(tempfile.mkdtemp(prefix='hyso-full-acceptance-regression-'))
    repository=Path(m['repository']);sys.path.insert(0,str(repository/'tests'))
    args=['-c',str(repository/'tests/pytest.ini'),'-q','--tb=short','--maxfail=1','--basetemp='+str(temp),
        '--junitxml='+str(root/'tests.xml'),*[str(repository/x) for x in SAFE_TESTS]]
    save(root,'selection',dict(files=SAFE_TESTS,temporary_directory=str(temp),native_GPU_enabled=False,
        note='No all-pytest collection; CUDA protocol doubles are not native GPU workloads.'))
    status=int(pytest.main(args));return result('PASS' if status==0 else 'FAIL',reason=None if status==0 else 'safe_regression_failure',
        evidence=[record(root/'tests.xml')],pytest_exit=status,criteria=dict(safe_regressions=status==0))

def inference_benchmark(model,sample,p,device):
    from hyso.benchmark import BenchmarkRunner
    if device!='cpu':
        import copy
        model=copy.deepcopy(model).to(device)
        sample=sample.to(device)
    return BenchmarkRunner(model,device=device,precision='fp32',example_input=sample,
        warmup_iterations=p['warmup'],measure_iterations=p['samples'],repeats=p['repeats'],seed=p['seed'],measure_cuda_memory=device!='cpu').run()

def cuda_benchmark_check(m,root):
    """One timed CUDA sample per arm; full acceptance protocol remains unchanged."""
    import torch
    from .workloads import configure,load_model,data
    from hyso._evaluation_evidence import model_hash
    from hyso.performance.devices import _cuda_hardware
    configure(m)
    if not torch.cuda.is_available():raise ValueError('native CUDA device unavailable; no fallback')
    torch.cuda.set_device(0)
    evidence=[save(root,'physical-GPU-runtime-identity',_cuda_hardware(torch,m['hardware']['gpu_uuid']))]
    protocol=dict(m['protocols']['inference'],samples=1)
    for arm in ('A','B'):
        model=load_model(m,arm);state=model_hash(model);x,_=data(m,count=protocol['batch'])
        evidence.append(save(root,arm+'-identity',dict(checkpoint=m['assets']['models'][arm],state_sha256=state,
            input_shape=list(x.shape),input_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
            precision=protocol['precision'],protocol=protocol,device='cuda:0',gpu_uuid=m['hardware']['gpu_uuid'])))
        measured=inference_benchmark(model,x,protocol,'cuda:0')
        evidence.append(save(root,arm+'-benchmark',measured))
        if model_hash(model)!=state:raise ValueError('CPU master state changed')
        if sha(m['assets']['models'][arm]['path'])!=m['assets']['models'][arm]['sha256']:raise ValueError('checkpoint changed')
        if any(t.device.type!='cpu' for t in [*model.parameters(),*model.buffers()]):raise ValueError('CPU master moved')
    return result(evidence=evidence,criteria=dict(resident_CUDA_boundary=True,CPU_master_preserved=True),
        acceptance_scope='short CUDA boundary only: A/B original batch256, FP32, warmup3/repeat1, one timed sample each plus runner preflight/memory probe; not 04.cuda acceptance')

def cpu_benchmark_check(m,root):
    """Bounded check of the failed constructor boundary, not 03.cpu acceptance."""
    import torch
    from .workloads import configure,load_model,data
    from hyso._evaluation_evidence import model_hash
    configure(m);evidence=[]
    for arm in ('A','B'):
        model=load_model(m,arm);state=model_hash(model);x,_=data(m,count=1)
        protocol=dict(warmup=1,samples=3,repeats=1,seed=m['protocols']['inference']['seed'])
        evidence.append(save(root,arm+'-benchmark',inference_benchmark(model,x,protocol,'cpu')))
        if model_hash(model)!=state or torch.cuda.is_initialized():raise ValueError('check changed model or initialized CUDA')
        evidence.append(save(root,arm+'-identity',dict(checkpoint=m['assets']['models'][arm],state_sha256=state,
            input_shape=list(x.shape),input_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),protocol=protocol)))
    return result(evidence=evidence,criteria=dict(constructor_boundary=True,source_state_preserved=True,CUDA_initialized=False),
        acceptance_scope='A/B CPU batch1: 1 warmup +3 timed forwards each, plus runner preflight; not full 03.cpu acceptance')

def inference(m,root,device):
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    from hyso.benchmark import BenchmarkRunner
    from hyso.performance import (analyze_latency,measure_throughput,ThroughputProtocol,measure_footprint,measure_memory,
        MemoryProtocol,run_sweep,SweepProtocol,PipelineSample,PipelineStage,PipelineProtocol,measure_pipeline,analyze_stages,
        measure_utilization,UtilizationProtocol,measure_transfers,TransferProtocol)
    from hyso import evaluate_classification,EvaluationCohort,evaluate_precision_quality,evaluate_robustness,compare_performance_quality
    from hyso._evaluation_evidence import model_hash
    from hyso.performance._batch_worker import _private_commit
    from .workloads import configure,load_model,data
    configure(m);p=m['protocols']['inference'];gpu=device!='cpu'
    if gpu:
        if not m['hardware']['gpu_uuid'] or not torch.cuda.is_available():return result('UNSUPPORTED',reason='native_CUDA_device_unavailable')
        torch.cuda.set_device(0)
        from hyso.performance.devices import _cuda_hardware
        save(root,'physical-GPU-runtime-identity',_cuda_hardware(torch,m['hardware']['gpu_uuid']))
    x,y=data(m);evidence=[];metrics=[];identities={};models={}
    cohort=EvaluationCohort(x,y,sample_ids=[f'mnist:test:{i:05}' for i in range(len(x))],class_names=[str(i) for i in range(10)],
        split='test',preprocessing_id=m['assets']['dataset']['preprocessing'])
    for arm in ('A','B'):
        folder=root/arm;folder.mkdir();model=load_model(m,arm);state=model_hash(model);sample=x[:p['batch']].clone()
        identity=dict(checkpoint=m['assets']['models'][arm],device=device,gpu_uuid=m['hardware']['gpu_uuid'] if gpu else None,
            precision='fp32',input_shape=list(sample.shape),input_sha256=hashlib.sha256(sample.numpy().tobytes()).hexdigest(),
            preprocessing=m['assets']['dataset']['preprocessing'],state_sha256=state,quality_samples=len(x),quality_batch=p['quality_batch'],
            cpu_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),protocol=p)
        save(folder,'identity',identity);identities[arm]=identity
        models[arm]=model
        b=inference_benchmark(model,sample,p,device)
        evidence.append(save(folder,'benchmark',b));evidence.append(save(folder,'latency',analyze_latency(b,quantiles=(.5,.95,.99))))
        import copy
        observed=copy.deepcopy(model).to(device);observed_x=sample.to(device)
        q=measure_throughput(observed,observed_x,protocol=ThroughputProtocol(device=device,warmup_iterations=p['warmup'],
            batch_count=p['completed_batches'],repeats=p['repeats']))
        evidence.append(save(folder,'throughput',q))
        def observed_step():
            with torch.inference_mode():output=observed(observed_x)
            if gpu:torch.cuda.synchronize(0)
            if not torch.isfinite(output).all().item():raise ValueError('utilization workload output invalid')
        running=threading.Event();errors=[];completed=[0]
        def drive():
            try:
                while not running.is_set():observed_step();completed[0]+=1
            except BaseException as exc:errors.append(exc);running.set()
        driver=threading.Thread(target=drive);driver.start()
        try:
            util=measure_utilization(protocol=UtilizationProtocol(duration_seconds=2.,interval_seconds=.1,
                gpu=gpu,gpu_selector=m['hardware']['gpu_uuid'] if gpu else '0',workload_label=m['assets']['models'][arm]['model_id']))
        finally:running.set();driver.join(60)
        if driver.is_alive():raise RuntimeError('owned utilization workload failed to close; no retry')
        if errors:raise errors[0]
        evidence.append(save(folder,'utilization',util));save(folder,'utilization-work',dict(completed_calls=completed[0],
            scope='same_process_independent_driver_thread;rate_not_substituted_for_P02;device_busy_not_model_exclusive'))
        del observed,observed_x
        quality=evaluate_classification(model,DataLoader(TensorDataset(x,y),batch_size=p['quality_batch'],shuffle=False),
            class_names=[str(i) for i in range(10)],split='test',device=device,source_id='full_acceptance:'+m['run_id']+':'+arm,
            capture_evidence=not gpu)
        evidence.append(save(folder,'quality',quality))
        for precision in [v for v in p['precisions_cuda' if gpu else 'precisions_cpu'] if v!='fp32']:
            evidence.append(save(folder,'quality-'+precision,evaluate_precision_quality(model,cohort,candidate=precision,device=device,batch_size=p['quality_batch'])))
        if not gpu:
            rx=x[:p['robustness_samples']];ry=y[:len(rx)]
            paired=EvaluationCohort(rx,ry,sample_ids=[f'mnist:test:{i:05}' for i in range(len(rx))],class_names=[str(i) for i in range(10)],
                split='test',preprocessing_id=m['assets']['dataset']['preprocessing'])
            def noise(t,g,severity):return (t+torch.randn(t.shape,generator=g,dtype=t.dtype)*severity).clamp(0,1)
            evidence.append(save(folder,'robustness',evaluate_robustness(model,paired,noise,corruption_id='gaussian_noise',severity=p['robustness_severity'],seed=p['seed'])))
        evidence.append(save(folder,'footprint',measure_footprint(model,artifact_path=m['assets']['models'][arm]['path'],
            artifact_format='full_training_checkpoint_not_weight_only',expected_sha256=m['assets']['models'][arm]['sha256'])))
        if not gpu:
            commit0=_private_commit()
            with torch.inference_mode():ram=measure_memory(lambda:model(sample),protocol=MemoryProtocol())
            evidence.append(save(folder,'process-memory',ram));commit1=_private_commit()
            metrics.extend([metric(arm+'.private_commit_before',commit0,'byte','measured','worker process, not model-only'),
                            metric(arm+'.private_commit_after',commit1,'byte','measured','worker process, not RSS')])
        def factory(shape,dtype,dev):
            images=sample[:shape[0]] if shape[0]<=len(sample) else sample.repeat((math.ceil(shape[0]/len(sample)),1,1,1))[:shape[0]]
            if tuple(images.shape[2:])!=tuple(shape[2:]):images=torch.nn.functional.interpolate(images,size=tuple(shape[2:]),mode='bilinear',align_corners=False)
            return images.to(dtype=dtype,device=dev)
        sweep=SweepProtocol(device=device,warmup_iterations=p['warmup'],measure_iterations=20,repeats=1,sustained_batches=10)
        evidence.append(save(folder,'batch-sweep',run_sweep(model,factory,axis='batch',points=p['sweep_batches'],protocol=sweep,input_shape=tuple(sample.shape))))
        evidence.append(save(folder,'resolution-sweep',run_sweep(model,factory,axis='resolution',points=[tuple(v) for v in p['resolution_points']],
            protocol=sweep,input_shape=(1,1,28,28))))
        evidence.append(save(folder,'precision-sweep',run_sweep(model,factory,axis='precision',points=p['precisions_cuda' if gpu else 'precisions_cpu'],
            protocol=sweep,input_shape=tuple(sample.shape))))
        ids=iter(range(1000))
        def source():
            i=next(ids);return PipelineSample(str(i),x[i:i+1].clone())
        def post(v):return PipelineSample(v.sample_id,v.payload.argmax(1).float().reshape(1,1,1,1))
        transfer=PipelineStage('transfer',lambda v:PipelineSample(v.sample_id,v.payload.to('cuda:0')),'cuda') if gpu else None
        pipeline_model=copy.deepcopy(model).to(device)
        pipeline=measure_pipeline(pipeline_model,source,transfer=transfer,postprocess=PipelineStage('postprocess',post,'cuda' if gpu else 'cpu'),
            protocol=PipelineProtocol(device=device,samples=40,repeats=1,warmup_samples=1))
        del pipeline_model
        evidence.append(save(folder,'pipeline',pipeline));evidence.append(save(folder,'stages',analyze_stages(pipeline)))
        if model_hash(model)!=state:raise ValueError('master model mutated')
        if sha(m['assets']['models'][arm]['path'])!=m['assets']['models'][arm]['sha256']:raise ValueError('checkpoint changed during inference')
        if any(t.device.type!='cpu' for t in [*model.parameters(),*model.buffers()]):raise ValueError('CPU master moved')
        metrics.append(metric(arm+'.quality_accuracy',quality.to_dict()['metrics']['accuracy'],'ratio','derived','same official10000 MNIST test cohort'))
    save(root,'model-identities',identities)
    if not gpu:
        for precision in ('fp32','fp64'):
            evidence.append(save(root,'A-B-tradeoff-'+precision,compare_performance_quality(models['A'],models['B'],cohort,
                precision=precision,iterations=20,batch_size=p['batch'],memory_metric='logical_model_bytes',independent_samples=False)))
    else:
        for pinned,nonblocking in ((False,False),(True,False),(True,True)):
            evidence.append(save(root,'transfer-'+str(pinned)+'-'+str(nonblocking),measure_transfers(protocol=TransferProtocol(
                shape=(p['batch'],1,28,28),warmup_iterations=p['warmup'],measure_iterations=20,repeats=1,
                pinned=pinned,non_blocking=nonblocking,timeout_seconds=120,cpu_memory_limit_bytes=None,input_limit_bytes=None,cuda_memory_fraction=None))))
    return result(criteria=dict(trained_A_B=True,matched_input_protocol=True,source_model_state_preserved=True),evidence=evidence,
        metrics=metrics,limits=['P05 cold start is retained evidence; no new cold-cache claim','P07 extra resolutions are declared new protocol, not copies of historical8-point acceptance',
            'P28 uses same-device fresh FP32 baselines; sweep latency is not quality','new P29 single-pair descriptions do not certify independent-trial significance or verified improvement'],device=device)

def training(m,root):
    from hyso.performance import (measure_training_with_loader,measure_gradient_diagnostics,TrainingMemoryRecipe,
        IsolatedTrainingMemoryProtocol,measure_training_memory_isolated)
    from .workloads import training_recipe
    p=m['protocols']['training'];kw=dict(manifest_path=str(Path(m['output'])/'manifest.json'))
    t=training_recipe(device='cpu',**kw)
    fit=measure_training_with_loader(t['trainer'],t['train_loader'],t['val_loader'],epochs=p['epochs'],stages=True)
    evidence=[save(root,'timing',fit.timing),save(root,'stages',fit.stages),save(root,'loader',fit.loader),save(root,'throughput',fit.throughput)]
    t=training_recipe(device='cpu',**kw)
    evidence.append(save(root,'gradient',measure_gradient_diagnostics(t['trainer'],t['train_loader'],t['val_loader'],epochs=p['epochs'])))
    recipe=TrainingMemoryRecipe('full_acceptance.workloads:training_recipe',kwargs=kw,device='cpu',epochs=p['epochs'],
        factory_search_path=str(Path(m['source_snapshot']['root'])/'tools'),seed=p['seed'])
    mem=measure_training_memory_isolated(recipe,protocol=IsolatedTrainingMemoryProtocol(timeout_seconds=600,
        process_memory_bytes=None,job_memory_bytes=None,max_processes=4,cuda_memory_fraction=None,saved_activation_lifetime=True))
    evidence.append(save(root,'isolated-memory',mem))
    from hyso import EvaluationCohort,measure_time_to_quality
    from .workloads import data
    import torch
    vx,vy=data(m,split='val',count=p['validation_samples'])
    import numpy as np
    with np.load(m['assets']['split_indices']['path'],allow_pickle=False) as split:
        idx=split['validation' if 'validation' in split.files else 'val'][:len(vx)]
    cohort=EvaluationCohort(vx,vy,sample_ids=['mnist:train:'+str(i) for i in idx],
        class_names=[str(i) for i in range(10)],split='validation',preprocessing_id=m['assets']['dataset']['preprocessing'])
    def target_factory(seed):
        built=training_recipe(device='cpu',**kw)
        from torch.utils.data import DataLoader,TensorDataset
        tensors=built['train_loader'].dataset.tensors
        loader=DataLoader(TensorDataset(*[v[:256] for v in tensors]),batch_size=p['batch'],shuffle=False,num_workers=0)
        return built['trainer'],loader
    evidence.append(save(root,'time-to-target',measure_time_to_quality(target_factory,cohort,seeds=p['target_seeds'],target=p['target_accuracy'],epochs=p['epochs'])))
    return result('PARTIAL',reason='CPU_current_source_evidence_only;CUDA_training_and_disabled_CUDA_saved_activation_not_claimed',
        criteria=dict(real_trained_B_CPU_training=True,distinct_observer_fits_separate=True,isolated_P23=True),evidence=evidence,
        metrics=[metric('epochs_per_distinct_fit',p['epochs'],'epoch','measured','three separate observer fits, not one merged timeline')])

def layer(m,root):
    from hyso.performance import (ColdStartRecipe,ProfilingProtocol,profile_operators,profile_layers,analyze_flops,
        LayerOverheadProtocol,measure_layer_overhead)
    p=m['protocols']['layer'];tools=Path(m['source_snapshot']['root'])/'tools'
    recipe=ColdStartRecipe('full_acceptance.workloads:model_only',model_kwargs=dict(manifest_path=str(Path(m['output'])/'manifest.json'),arm='B'),
        input_shape=(1,1,28,28),device='cpu',factory_search_path=str(tools))
    capture=ProfilingProtocol(warmup_iterations=p['warmup'],measure_iterations=p['windows'],timeout_seconds=120,
        profile_memory=True,allocation_lifetimes=True,cpu_memory_ownership=True)
    prof=profile_operators(recipe,protocol=capture,trace_path=root/'trained-trace.json')
    refs=[save(root,'operator-profile',prof),save(root,'layers',profile_layers(prof)),save(root,'operations',analyze_flops(prof))]
    shared=ColdStartRecipe('cpu_layer_factories:Shared',input_shape=(1,1,2,8),device='cpu',
        factory_search_path=str(Path(m['source_snapshot']['root'])/'tests'))
    overhead=measure_layer_overhead(shared,protocol=LayerOverheadProtocol(trials=p['trials'],priming_iterations=p['priming'],
        max_coefficient_of_variation=p['max_cv'],max_relative_half_width=p['max_relative_half_width']),capture_protocol=capture)
    refs.append(save(root,'ownership-calibration',overhead));d=overhead.to_dict()
    if d['measurements']['status']!='estimated':return result('FAIL',reason='ownership_calibration_screen_failed: '+str(d['measurements'].get('reason')),
        evidence=refs,criteria=dict(real_profiler=True,ownership_estimate=False))
    return result('PARTIAL',reason='CPU_ownership_calibration_does_not_establish_CUDA_module_kernel_attribution',
        evidence=refs,criteria=dict(real_profiler=True,ownership_estimate=d['measurements'].get('status')=='estimated'),
        metrics=[metric('calibration_pair_count',p['trials'],'pair','measured','fresh control/instrumented pairs; windows not independent pairs')])

def roofline(m,root):
    from hyso.performance.roofline_counters import read_ncu_counters,analyze_counter_roofline,_sample,_summary
    ncu=m['optional']['ncu_plan']['configuration']['tool'];script=Path(m['source_snapshot']['root'])/'tools/full_acceptance/counter_worker.py'
    if sha(ncu['path'])!=ncu['sha256']:raise ValueError('NCU tool changed')
    metrics=['gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum',
        'smsp__sass_thread_inst_executed_op_fadd_pred_on.sum','smsp__sass_thread_inst_executed_op_fmul_pred_on.sum',
        'smsp__sass_thread_inst_executed_op_ffma_pred_on.sum','profiler__replayer_passes',
        'sm__cycles_elapsed.avg.per_second','dram__cycles_elapsed.avg.per_second',
        'device__attribute_pci_domain_id','device__attribute_pci_bus_id','device__attribute_pci_device_id','device__attribute_display_name']
    sessions={};refs=[];commands=[]
    for role in ('target','compute','memory'):
        folder=root/role;folder.mkdir()
        cmd=[ncu['path'],'--devices','0','--target-processes','application-only','--profile-from-start','off','--nvtx','--nvtx-include',
            'hyso_p18_roof_'+role+'/','--launch-count','3','--kill','no','--replay-mode','kernel','--clock-control','none',
            '--cache-control','all','--check-exit-code','yes','--metrics',','.join(metrics),'--page','raw','--csv','--print-units','base',
            '--export',str(folder/'capture'),m['runtime']['python']['path'],'-B',str(script),'--manifest',str(Path(m['output'])/'manifest.json'),
            '--case',role,'--output',str(folder)]
        commands.append(cmd);save(root,'commands-through-'+role,commands)
        with (folder/'raw.csv').open('xb') as out,(folder/'stderr.log').open('xb') as err:
            p=subprocess.run(cmd,stdout=out,stderr=err,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
        if p.returncode:raise RuntimeError('NCU capture failed, no retry; exit='+str(p.returncode))
        lines=(folder/'raw.csv').read_text(encoding='utf-8-sig').splitlines();start=next(i for i,l in enumerate(lines) if l.startswith(('"ID",','ID,')))
        parsed=list(csv.DictReader(lines[start:]));names={row['Kernel Name'] for row in parsed[1:]}
        if len(names)!=1:raise ValueError('ambiguous counter kernel')
        session=read_ncu_counters(folder/'raw.csv',receipt_path=folder/'capture.json',session_id=m['run_id']+':'+role,
            role=role,kernel_name=next(iter(names)),source_paths={'worker':script})
        sessions[role]=session;refs.append(save(folder,'session',session))
    summary=_summary(sessions['compute']);spread=summary['range_over_median']*100
    independent=[_sample(s)['gflops_per_second'] for s in sessions['compute']['samples']]
    recomputed=100*(max(independent)-min(independent))/statistics.median(independent)
    if not math.isclose(spread,recomputed,rel_tol=1e-12):raise ValueError('independent calibration arithmetic mismatch')
    source=read(root/'compute/capture.json');expected=source['reference']['expected_ffma_thread_instructions']
    if any(s['counters']['smsp__sass_thread_inst_executed_op_ffma_pred_on.sum']['value']!=expected for s in sessions['compute']['samples']):
        raise ValueError('actual FFMA count differs from independent source reference')
    ok=spread<=10.;save(root,'calibration-screen',dict(status='PASS' if ok else 'FAIL',limit_percent=10.,observed_percent=spread,
        original_failed_session_retained=True,physical_cause='not_determined_by_three_samples',criterion_unchanged=True))
    rr=analyze_counter_roofline(sessions['target'],compute_sessions=[sessions['compute']],memory_session=sessions['memory'])
    refs.append(save(root,'roofline',rr))
    return result('PASS' if ok else 'FAIL',reason=None if ok else 'original_10_percent_compute_calibration_screen_failed',evidence=refs,
        criteria=dict(P18_calibration_screen=ok,independent_counter_work_reference=True),
        metrics=[metric('compute_spread',spread,'percent','derived','three independent new reference samples; original10% criterion')],
        limits=['controlled SIMT/vector-add references are not trained-CNN DRAM evidence','no stable hardware capacity, causal bottleneck or speedup claim'])

def stability(m,root,*,energy=False):
    from hyso.performance import WorkloadRecipe,StabilityProtocol,measure_stability,reopen_stability,analyze_energy,RuntimeResult
    mp=Path(m['output'])/'manifest.json';device=m['hardware']['gpu_uuid'] if energy else 'cpu'
    recipe=WorkloadRecipe('full_acceptance.workloads:inference_work',dict(manifest_path=str(mp),device=device),
        search_path=str(Path(m['source_snapshot']['root'])/'tools'),gpu_uuid=None if device=='cpu' else device)
    if energy:
        p=m['protocols']['energy'];protocol=StabilityProtocol(duration_seconds=p['active_seconds'],interval_seconds=p['interval_seconds'],
            idle_seconds=p['idle_seconds'],startup_timeout_seconds=120,step_timeout_seconds=60,max_samples=2000,chunk_rows=1,
            process_memory_bytes=None,job_memory_bytes=None,max_processes=None)
    else:protocol=StabilityProtocol(**m['protocols']['stability'])
    raw=measure_stability(recipe,root/'journal',protocol=protocol);refs=[save(root,'stability',raw)]
    d=raw.to_dict();diagnostics=raw.stability_diagnostics();save(root,'diagnostics',diagnostics)
    if reopen_stability(root/'journal').to_dict()['evidence']!=d['evidence']:raise ValueError('journal reopen mismatch')
    a=d['analysis'];c=d['evidence']['controller'];terminal=d['evidence']['terminal']
    complete=a['status']=='completed' and c['exit_code']==0 and c['cleanup']['active_processes']==0 and terminal is not None and terminal['closed']
    actual=diagnostics.get('completion_window',{}).get('seconds')
    if actual is None:actual=a.get('duration_seconds')
    complete=complete and actual is not None and actual>=protocol.duration_seconds
    metrics=[metric('actual_work_seconds',actual,'s','derived','one worker actual completed work window'),
        metric('completed_images',a['completed'],'image','measured','same source completion ledger')]
    if energy:
        en=analyze_energy(raw,max_gap_seconds=p['max_gap_seconds'],idle_max_age_seconds=p['idle_max_age_seconds']);refs.append(save(root,'energy',en));ea=en.to_dict()['analysis']
        metrics.extend([metric('gross_J',ea['gross_joules'],'J','derived','same UUID board power/work window',ea['reason'] or 'gross unavailable'),
            metric('net_J',ea['net_above_idle_joules'],'J','derived','inactive baseline subtracted; not model-exclusive',ea.get('idle_baseline',{}).get('reason') or 'net unavailable')])
        ok=complete and ea['gross_joules'] is not None and ea['net_above_idle_joules'] is not None and ea['source_complete']
        return result('PASS' if ok else 'PARTIAL',reason=None if ok else 'incomplete_matching_power_idle_work_coverage',
            evidence=refs,metrics=metrics,criteria=dict(trained_energy_full=ok),limitations=ea['limitations'])
    ok=complete and actual>=7200 and diagnostics['continuity']['requested_duration_satisfied'] and diagnostics['continuity']['no_failures_observed'] and diagnostics['continuity']['no_sampling_gaps_detected']
    return result('PASS' if ok else 'FAIL',reason=None if ok else 'long_stability_criterion_not_met',evidence=refs,metrics=metrics,
        criteria=dict(P31_uninterrupted_7200=ok,no_sampling_gaps=diagnostics['continuity']['no_sampling_gaps_detected']),
        diagnostic_scope='latency/throughput drift and RSS/private commit from this single work series; no causal leak guarantee')

def devices(m,root):
    from hyso.performance import WorkloadRecipe,DeviceProtocol,compare_devices
    refs=[]
    for arm in ('A','B'):
        recipe=WorkloadRecipe('full_acceptance.workloads:model_pair',dict(manifest_path=str(Path(m['output'])/'manifest.json'),arm=arm),
            search_path=str(Path(m['source_snapshot']['root'])/'tools'))
        p=m['protocols']['device']
        d=compare_devices(recipe,['cpu',m['hardware']['gpu_uuid']],root/(arm+'-workers'),protocol=DeviceProtocol(**p,
            startup_timeout_seconds=120,step_timeout_seconds=60,max_runtime_seconds=300,process_memory_bytes=None,job_memory_bytes=None,max_processes=None))
        refs.append(save(root,arm+'-comparison',d));a=d.to_dict()['analysis']
        if not a['comparable'] or any(row['identity']['origin']['kind']!='trained_checkpoint' for row in a['devices']):
            return result('FAIL',reason='trained_device_pair_not_comparable',evidence=refs)
        save(root,arm+'-diagnostics',d.device_diagnostics())
    return result(criteria=dict(trained_matched_CPU_GPU=True),evidence=refs,
        limits=['ratios belong only to this matched protocol','RSS, CUDA allocated/reserved, logical bytes remain separate','no universal winner or statistical superiority'])

def memory(m,root):
    from hyso.performance import RuntimeResult,analyze_memory_trend
    source=Path(m['output'])/'stages/10.stability/stability.json';raw=RuntimeResult.from_json(source)
    refs=[]
    for warmup in (0.,60.,300.):refs.append(save(root,'growth-warmup-'+str(int(warmup)),analyze_memory_trend(raw,warmup_seconds=warmup)))
    return result(evidence=refs,criteria=dict(actual_long_run_series=True),
        metrics=[metric('causal_leak_or_leak_absence',None,'claim','missing','descriptive memory trends','no ownership/heap/storage proof; long stability is not causal leak absence')])

def camera(m,root,*,live):
    from hyso.performance import WorkloadRecipe,StreamSource,StreamProtocol,measure_streaming
    cfg=m['optional']['camera_adapter']['configuration'];source=StreamSource('camera' if live else 'video_file',
        cfg.get('camera_index',0) if live else cfg['video']['path'],cfg['provenance'])
    recipe=WorkloadRecipe('full_acceptance.workloads:frame_processor',dict(manifest_path=str(Path(m['output'])/'manifest.json')),
        search_path=str(Path(m['source_snapshot']['root'])/'tools'))
    p=m['protocols']['camera' if live else 'video'];raw=measure_streaming(source,recipe,root/'journal',protocol=StreamProtocol(**p,
        max_runtime_seconds=180,startup_timeout_seconds=120,heartbeat_timeout_seconds=60))
    ref=save(root,'stream',raw);d=raw.to_dict();a=d['analysis']
    conservation=a['received']==a['processed']+a['dropped']+a['pending'];ok=a['status']=='completed' and conservation
    return result('PASS' if ok else 'FAIL',evidence=[ref],criteria=dict(real_camera_trained=bool(live and ok),conservation=conservation),
        metrics=[metric('processed_fps',a['completed_processing_fps'],'frame/s','derived','host application window; not nominal sensor FPS')],
        limits=['sensor/exposure/upstream driver drops unavailable unless adapter measures them','classification correctness needs its own appropriate labeled camera dataset'])

def provider(m,root,*,cuda):
    from hyso.performance import WorkloadRecipe,ProviderProtocol,profile_execution_providers
    mp=str(Path(m['output'])/'manifest.json');tools=str(Path(m['source_snapshot']['root'])/'tools');refs=[]
    protocol=ProviderProtocol(warmup=3,runs=10,startup_timeout_seconds=120,step_timeout_seconds=60,max_runtime_seconds=300,
        process_memory_bytes=None,job_memory_bytes=None,max_processes=None)
    if not cuda:
        for arm in ('A','B'):
            recipe=WorkloadRecipe('full_acceptance.workloads:onnx_trained',dict(manifest_path=mp,arm=arm,export_path=str(root/(arm+'.onnx'))),search_path=tools)
            value=profile_execution_providers(recipe,root/(arm+'-worker'),protocol=protocol,providers=['CPUExecutionProvider'],expected={'*':'CPUExecutionProvider'},
                optimization='disabled',runtime_search_path=m['hardware']['ort_cpu'])
            refs.append(save(root,arm+'-provider',value));diag=value.provider_diagnostics();save(root,arm+'-diagnostics',diag)
            if not diag['events'] or value.to_dict()['analysis']['status']!='completed':return result('FAIL',reason='no_real_trained_ORT_execution',evidence=refs)
    else:
        tests=str(Path(m['source_snapshot']['root'])/'tests')
        for name,factory in (('relu','deployment_support:onnx_model'),('mixed','provider_acceptance_factories:partition_model')):
            recipe=WorkloadRecipe(factory,search_path=tests,gpu_uuid=m['hardware']['gpu_uuid'])
            value=profile_execution_providers(recipe,root/(name+'-worker'),protocol=protocol,
                providers=['CUDAExecutionProvider','CPUExecutionProvider'],expected={'*':'CUDAExecutionProvider'},optimization='disabled',
                runtime_search_path=m['hardware']['ort_cuda'],provider_options={'CUDAExecutionProvider':{'device_id':0}})
            refs.append(save(root,name+'-provider',value));diag=value.provider_diagnostics();save(root,name+'-diagnostics',diag)
            observed=set(diag['observed_execution_providers'])
            if 'CUDAExecutionProvider' not in observed or name=='mixed' and observed!={'CUDAExecutionProvider','CPUExecutionProvider'}:
                return result('FAIL',reason='expected_actual_native_node_placement_missing',evidence=refs)
    return result(evidence=refs,criteria=dict(actual_node_execution=True),limits=['availability is not execution','mixed controlled graph is not trained A/B mixed fallback','no execution-error retry recovery claim'])

def replay(m,root):
    import torch
    from unittest.mock import patch
    from contextlib import ExitStack
    from hyso.analysis._native import validate_native
    from hyso.performance import IsolatedTrainingMemoryResult
    from hyso.training import Trainer
    from hyso.benchmark import BenchmarkRunner
    rows=read(m['prior_audit']['replay-inputs.json']['path'])['checks'];checks=[];forbidden=[]
    def guard(label):
        def denied(*a,**k):forbidden.append(label);raise AssertionError('offline-only '+label)
        return denied
    with ExitStack() as stack:
        for obj,name in ((torch.nn.Module,'_call_impl'),(torch.Tensor,'backward'),(torch,'load'),(Trainer,'fit'),(BenchmarkRunner,'run'),(subprocess,'Popen')):
            stack.enter_context(patch.object(obj,name,guard(name)))
        for name in ('init','_lazy_init','is_available','device_count','current_device','synchronize','set_device'):
            stack.enter_context(patch.object(torch.cuda,name,guard('cuda.'+name)))
        for r in rows:
            path=Path(m['repository'])/r['path']
            if sha(path)!=r['artifact_sha256']:raise ValueError('replay source changed')
            d=read(path)
            for key in r['json_pointer'].split('/')[1:]:d=d[int(key)] if isinstance(d,list) else d[key]
            try:
                if r['format']=='hyso.performance.isolated_training_memory':IsolatedTrainingMemoryResult(d)
                else:validate_native(d)
            except Exception as error:
                if not (r['expected']=='REJECT' and type(error).__name__=='ValidationError' and 'MemcpyToHost' in str(error)):raise
                status='EXPECTED_REJECTION'
            else:
                if r['expected']=='REJECT':raise AssertionError('invalid draft unexpectedly accepted')
                status='VALIDATED'
            checks.append(dict(path=str(path),pointer=r['json_pointer'],payload_sha256=r['payload_sha256'],status=status))
    if forbidden:raise AssertionError('offline guard calls')
    ref=save(root,'saved-reader-checks',dict(checks=checks,unique_payloads=len({r['payload_sha256'] for r in checks}),forbidden_calls=forbidden))
    return result(evidence=[ref],criteria=dict(evidence_replay=True),metrics=[metric('artifact_checks',len(checks),'check','measured','duplicates are not independent experiments')])
