"""Explicit acceptance experiments; no arbitrary command or plugin execution."""

STAGES=(
 ('01','Source/checkpoint/dataset/environment preflight'),
 ('02','Safe regression and compatibility'),('03','Trained CPU inference'),
 ('04','Trained CUDA inference'),('05','Training and training memory'),
 ('06','Layer profiling and roofline calibration'),('07','Energy and device comparison'),
 ('08','Camera/video'),('09','ONNX Runtime provider/fallback'),
 ('10','Continuous7200-second trained stability'),('11','Memory-growth analysis'),
 ('12','Evidence replay and acceptance reconciliation'),
)

# Authorization and prerequisite checks happen independently for each experiment.
# No implicit authorization through a parent stage, previous task or preset.
EXPERIMENTS=(
 ('01.preflight','01','environment',True,(),(), 'metadata only; no CUDA context, model or sensor acquisition'),
 ('02.regression','02','regression',True,(),(), 'explicit reviewed CPU test selection, existing skips retained'),
 ('03.cpu','03','inference_cpu',True,('02.regression',),(), 'trained A/B CPU FP32; same declared resident batch and full test cohort'),
 ('04.cuda','04','inference_cuda',False,('03.cpu',),('gpu',), 'trained A/B native CUDA; no allocator fraction'),
 ('05.training','05','training',False,('02.regression',),(), 'fresh trained Model B clones; real train/validation split, separate fits'),
 ('06.layer','06','layer',False,('02.regression',),(), 'real CPU ownership profile and5 paired calibration workers; no B4 replay'),
 ('06.roofline','06','roofline',False,('02.regression',),('ncu_plan',), 'independent native counter reference sessions; original10% spread screen'),
 ('07.energy','07','energy',False,('04.cuda',),('gpu',), 'trained Model B same UUID idle+active physical board samples'),
 ('07.devices','07','devices',False,('04.cuda',),('gpu',), 'matched trained A/B CPU/GPU source protocol; no cross-run substitution'),
 ('08.video','08','video',False,('02.regression',),('camera_adapter','video'), 'existing real video with explicit task-appropriate trained adapter'),
 ('08.camera','08','camera',False,('02.regression',),('camera_adapter',), 'live capture; MNIST is not assumed appropriate'),
 ('09.provider_cpu','09','provider_cpu',False,('03.cpu',),('ort_cpu',), 'new trained ONNX export + actual CPU node events'),
 ('09.provider_cuda','09','provider_cuda',False,('09.provider_cpu',),('gpu','ort_cuda'), 'actual CUDA/mixed controlled ONNX node placement; not trained mixed acceptance'),
 ('10.stability','10','stability',False,('03.cpu',),(), 'Model B CPU FP32 batch256 uninterrupted actual work>=7200s'),
 ('11.memory','11','memory',True,('10.stability',),(), 'pure analysis of the new long-run raw series; no second workload'),
 ('12.replay','12','replay',True,(),(), 'hash/read-only old+new canonical native readers; reconciliation'),
)

SAFE_TESTS=[
 'tests/test_benchmark_result.py','tests/test_benchmark_comparison.py',
 'tests/test_benchmark_cuda_protocol.py','tests/test_resolution_sweep_cpu_oom.py',
 'tests/test_training_throughput_protocol.py','tests/test_isolated_training_memory_reader.py',
 'tests/test_p31_trained_launcher.py','tests/test_p32_completion.py','tests/test_p33_completion.py',
 'tests/test_analysis_evidence.py','tests/test_analysis_report.py',
]
FEATURE_EXPERIMENTS={
 **{f'P{i:02}':['03.cpu','04.cuda'] for i in range(1,13)},
 'P12':['06.layer'],'P13':['06.layer'],'P14':['06.layer','12.replay'],'P15':['04.cuda','12.replay'],
 'P16':['03.cpu','04.cuda','10.stability'],'P17':['12.replay'],
 'P18':['06.roofline'],'P19':['05.training'],'P20':['05.training'],'P21':['05.training'],
 'P22':['05.training'],'P23':['05.training'],'P24':['05.training'],'P25':['05.training'],
 'P26':['03.cpu','04.cuda'],'P27':['03.cpu'],'P28':['03.cpu','04.cuda','12.replay'],
 'P29':['03.cpu','12.replay'],'P30':['08.video','08.camera'],'P31':['10.stability'],
 'P32':['11.memory'],'P33':['07.energy'],'P34':['07.devices'],
 'P35':['09.provider_cpu','09.provider_cuda'],
}

def experiments():
    result=[]
    for ident,stage,handler,approved,depends,requires,scope in EXPERIMENTS:
        result.append(dict(id=ident,stage=stage,handler=handler,required=True,depends=list(depends),requires=list(requires),
            scope=scope,authorization=dict(approved=approved,
                basis='user request permits bounded CPU/offline execution; explicit --execute still required' if approved else 'NOT_AUTHORIZED: first delivery explicitly excludes this native experiment',
                approved_at=None),automatic_retry=False))
    return result
