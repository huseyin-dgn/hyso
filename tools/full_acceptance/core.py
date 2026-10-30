"""Strict stdlib manifests, content identities and exclusive source snapshots."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,sys,uuid
from .registry import STAGES,experiments,SAFE_TESTS

SCHEMA=1
ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'outputs/global-integration-audit-20261030'
DEFAULT_PYTHON=Path('C:/Users/hdgn5/.conda/envs/torch_gpu/python.exe')
TOOLING=[Path('C:/Users/hdgn5/AppData/Local/Temp/hyso-p01b-tests-40awqikl/Lib/site-packages'),
         Path('C:/Users/hdgn5/AppData/Roaming/Python/Python310/site-packages')]

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
def digest(obj):return hashlib.sha256(canonical(obj)).hexdigest()
def _unique(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise ValueError('duplicate JSON key: '+k)
        result[k]=v
    return result
def read(path):
    path=Path(path)
    if path.stat().st_size>256*1024**2:raise ValueError('bounded JSON exceeded')
    return json.loads(path.read_text(encoding='utf-8-sig'),object_pairs_hook=_unique,
                      parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite JSON: '+value)))
def publish(path,obj,*,new=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if new:
        with path.open('x',encoding='utf-8',newline='\n') as f:f.write(raw)
    else:
        tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        with tmp.open('x',encoding='utf-8',newline='\n') as f:f.write(raw)
        os.replace(tmp,path)
def record(path):
    p=Path(path).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf-8',errors='replace')

def source_files(root=None):
    root=Path(root or ROOT)
    paths=[]
    for folder in ('hyso','tools','tests','docs'):
        for p in (root/folder).rglob('*'):
            if not p.is_file() or any(x in p.parts for x in ('__pycache__','.pytest_cache','build','dist')):continue
            if p.suffix.lower() not in ('.py','.json','.md','.txt','.cfg','.ini','.toml','.cu','.cpp','.h','.ptx'):continue
            paths.append(p)
    for name in ('setup.cfg','pyproject.toml','README.md','LICENSE','.gitattributes','.gitignore'):
        p=root/name
        if p.is_file():paths.append(p)
    return sorted(set(paths))

def dataset_inventory():
    previous=read(ROOT/'outputs/real-two-model-comparison-20261029/protocol.json')
    protocol=read(ROOT/'outputs/p29-tradeoff-completion-20261029/protocol.json')
    md=read(previous['dataset_manifest']);models={}
    for arm in ('A','B'):
        item=protocol['models'][arm];cp=record(item['checkpoint'])
        if cp['sha256']!=item['checkpoint_sha256']:raise ValueError('trained checkpoint changed: '+arm)
        models[arm]=dict(**cp,model_id='trained_Model_'+arm,config=previous['models'][arm]['config'],
            recorded_state_sha256=item['state_sha256'],parameters=item['parameters'],completed_epochs=6)
    dataset=record(md['source_path'])
    if dataset['sha256']!=protocol['dataset_sha256']:raise ValueError('MNIST changed')
    data28=read(ROOT/'outputs/p28-precision-quality-completion-20261029/workers/cpu/data-identity.json')
    return dict(models=models,dataset=dict(**dataset,split='official_test',sample_count=10000,
        preprocessing=md['preprocessing'],cohort_sha256=data28['cohort_sha256'],ordered_stream=data28['stream']),
        split_indices=record(previous['split_indices']),split_manifest=record(previous['dataset_manifest']))

def runtime_inventory(python):
    python=Path(python).resolve()
    site=python.parent/'Lib/site-packages';paths=[site,*[p for p in TOOLING if p.is_dir()]]
    files=[python,site/'torch/_C.cp310-win_amd64.pyd',site/'torch/lib/torch_cpu.dll',
        site/'torch/lib/torch_cuda.dll',site/'torch/version.py',site/'torchvision/version.py']
    return dict(python=record(python),site_paths=[str(p) for p in paths],
        runtime_files=[record(p) for p in files if p.is_file()],
        expected=dict(torch='2.5.1+cu121',torchvision='0.20.1+cu121',python_major_minor='3.10'),
        tooling='existing local packages only; no install performed')

def prepare(output,*,python=DEFAULT_PYTHON,gpu_uuid=None,camera_config=None,ncu_plan=None,ort_cpu=None,ort_cuda=None):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists() or (out/'source').exists():raise FileExistsError('use a new output directory; manifests are never overwritten')
    baseline=out/'preservation-before.json'
    if not baseline.exists():
        rows=[]
        for base,dirs,names in os.walk(ROOT):
            dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','.pytest_cache') and Path(base,d).resolve()!=out]
            for name in names:
                p=Path(base,name);row=dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size)
                try:row['sha256']=sha(p)
                except OSError as error:row.update(sha256=None,read_error=str(error))
                rows.append(row)
        publish(baseline,dict(head=git('rev-parse','HEAD').strip(),status=git('status','--porcelain=v1','--untracked-files=all'),
            files=rows,excluded_output_directory=out.relative_to(ROOT).as_posix()),new=True)
    baseline_data=read(baseline)
    exclusions=[str(out)]
    if baseline_data.get('excluded_output_directory'):exclusions.append(str(ROOT/baseline_data['excluded_output_directory']))
    inventory=dataset_inventory();runtime=runtime_inventory(python)
    snapshot=out/'source';snapshot.mkdir();files={}
    for p in source_files():
        rel=p.relative_to(ROOT);dst=snapshot/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dst);files[rel.as_posix()]=sha(dst)
        if sha(p)!=files[rel.as_posix()]:raise ValueError('source changed while snapshotting')
    external={}
    for key in ('P01-P35-acceptance-matrix.json','replay-inputs.json','test-inventory.json'):
        external[key]=record(AUDIT/key)
    replay=read(AUDIT/'replay-inputs.json')
    replay_sources={}
    for row in replay['checks']:
        path=(ROOT/row['path']).resolve()
        if sha(path)!=row['artifact_sha256']:raise ValueError('replay artifact changed: '+str(path))
        replay_sources[str(path)]=row['artifact_sha256']
    extras={}
    for key,val in (('camera_adapter',camera_config),('ncu_plan',ncu_plan)):
        if val:extras[key]=dict(file=record(val),configuration=read(val))
    if ncu_plan is None:
        ncu=Path('C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.3.1/target/windows-desktop-win7-x64/ncu.exe')
        if ncu.is_file():extras['ncu_plan']=dict(configuration=dict(tool=record(ncu),protocol='fresh_target_compute_memory_three_samples_each',
            spread_limit_percent=10.,source='bundled counters source/PTX; no old measurements reused'))
    if gpu_uuid is None:
        previous_gpu=ROOT/'outputs/p28-precision-quality-completion-20261029/gpu-plan.json'
        if previous_gpu.is_file():
            d=read(previous_gpu)
            gpu_uuid=d.get('gpu_uuid') or d.get('GPU_UUID') or d.get('GPU_uuid')
    if gpu_uuid is not None:
        if not gpu_uuid.startswith('GPU-'):raise ValueError('explicit physical GPU UUID required')
        uuid.UUID(gpu_uuid[4:])
    if ort_cpu is None:
        old=ROOT/'outputs/p35-provider-fallback-20261030/saved-native-cpu/p35-cpu.json'
        if old.is_file():ort_cpu=read(old)['evidence'].get('runtime_search_path')
    if ort_cuda is None:
        old=Path('C:/Users/hdgn5/hyso-p35-native-evidence-20261027/cuda-relu.json')
        if old.is_file():ort_cuda=read(old)['evidence'].get('runtime_search_path')
    optional_runtime=[]
    for location in (ort_cpu,ort_cuda):
        if location:
            for p in Path(location,'onnxruntime').rglob('*'):
                if p.is_file() and p.suffix.lower() in ('.py','.pyd','.dll'):optional_runtime.append(record(p))
    runtime['optional_runtime_files']=optional_runtime
    exp=experiments()
    budgets={'environment':180,'regression':900,'inference_cpu':900,'inference_cuda':900,'training':1800,'layer':1500,
        'roofline':600,'energy':300,'devices':1500,'video':330,'camera':330,'provider_cpu':900,'provider_cuda':900,
        'stability':7440,'memory':300,'replay':900}
    for item in exp:
        item.update(output=str(out/'stages'/item['id']),device='cpu',model='A/B' if item['id'] in ('03.cpu','04.cuda','07.devices') else 'B',
            expected_work='see protocol and handler scope',stop_rules=['user_stop_request','Windows_OS_low_memory','source_or_asset_mismatch',
                'native_worker_crash_stop_entire_suite','heartbeat_stall','finite_evidence_budget_exhausted'],
            cleanup='terminate/reap only owned Windows Job tree; active_processes must be0; no automatic retry')
        item['outer_seconds']=budgets[item['handler']]
        item['deadline_basis']='native producer maximum +120s startup and cleanup reserve; finite work counts fixed below; no hardware capacity inference'
        if item['handler']=='stability':item['deadline_basis']='existing reviewed P31 plan:7200s work +120s startup +120s cleanup/export reserve =7440s'
        if item['handler'] in ('inference_cuda','energy','devices','provider_cuda','roofline'):item['device']=gpu_uuid
        if item['handler']=='stability':item['expected_work']='one Model B CPU FP32 batch256 actual uninterrupted7200s; no prefix concatenation'
        if item['handler'] in ('inference_cpu','inference_cuda'):item['expected_work']='each A/B:1000 latency forwards batch256 +40 completed throughput batches +10000-sample baseline/precision quality; declared finite sweeps/pipeline; CPU also paired robustness256 and matched P29 descriptions'
        if item['handler']=='energy':item['expected_work']='5s inactive baseline +30s completed trained Model B CUDA inference; physical power samples every.5s'
        if item['handler']=='devices':item['expected_work']='A and B separately:CPU and physical GPU each3 warmups+1000 timed/completed forwards batch256'
        if item['handler']=='layer':item['expected_work']='trained B CPU trace2 warmups+5 measured; shared ownership5 pairs x2 arms x(2warmup+2priming+5analyzed) forwards'
        item['command']=[runtime['python']['path'],'-I','-S','-B',str(snapshot/'tools/run_full_acceptance.py'),
            '--worker',item['id'],'--manifest',str(out/'manifest.json'),'--stage-output',item['output']]
    manifest=dict(format='hyso.full_acceptance.manifest',schema_version=SCHEMA,run_id=str(uuid.uuid4()),prepared_at=now(),
        repository=str(ROOT),output=str(out),head=git('rev-parse','HEAD').strip(),dirty_status=git('status','--porcelain=v1','--untracked-files=all'),
        source_snapshot=dict(root=str(snapshot),files=files,tree_sha256=digest(files),live_sources_must_match=True),
        assets=inventory,runtime=runtime,prior_audit=external,replay_sources=replay_sources,
        preservation_baseline=record(baseline),preservation_excluded_directories=sorted(set(exclusions)),
        external_preservation_baseline=record(out/'external-preservation-before.json') if (out/'external-preservation-before.json').exists() else None,
        stages=[dict(id=i,name=n) for i,n in STAGES],
        experiments=exp,optional=extras,hardware=dict(gpu_uuid=gpu_uuid,ort_cpu=str(Path(ort_cpu).resolve()) if ort_cpu else None,
            ort_cuda=str(Path(ort_cuda).resolve()) if ort_cuda else None),
        protocols=dict(inference=dict(batch=256,warmup=3,samples=1000,repeats=1,completed_batches=40,precision='fp32',
            preprocessing=inventory['dataset']['preprocessing'],cpu_threads=2,interop_threads=1,seed=20261030,
            timing='resident forward + device completion; setup/transfers excluded; host perf_counter_ns and CUDA event samples separate',
            quality_batch=256,quality_samples=10000,robustness_samples=256,robustness_severity=.05,sweep_batches=[1,8,32],resolution_points=[[28,28],[32,32],[28,36]],precisions_cpu=['fp32','fp64','bf16'],precisions_cuda=['fp32','amp_fp16','bf16']),
            training=dict(epochs=2,train_samples=512,validation_samples=128,batch=64,seed=20261030,device='cpu',AMP=False,
                description='fresh6-epoch trained Model B clone for each distinct observer fit; no original checkpoint writes',target_seeds=[11,29],target_accuracy=.99),
            layer=dict(trials=5,warmup=2,priming=2,windows=5,max_cv=.25,max_relative_half_width=.5),
            energy=dict(idle_seconds=5,active_seconds=30,interval_seconds=.5,max_gap_seconds=2.,idle_max_age_seconds=60.),
            device=dict(warmup=3,samples=1000,seed=20261030,precision='float32'),
            stability=dict(duration_seconds=7200.,interval_seconds=5.,idle_seconds=0.,startup_timeout_seconds=120.,
                step_timeout_seconds=60.,max_samples=2000,chunk_rows=1,max_journal_bytes=128*1024**2,
                seed=0,process_memory_bytes=None,job_memory_bytes=None,max_processes=None),
            video=dict(max_frames=120,queue_capacity=2,queue_policy='block'),camera=dict(max_frames=120,queue_capacity=2,queue_policy='drop_newest')),
        supervision=dict(startup_seconds=120,heartbeat_seconds=120,poll_seconds=.25,cleanup_seconds=15,
            justification='startup from reviewed P31 maximum; liveness heartbeat is emitted by a stdlib thread; actual native steps use producer deadlines',
            max_log_bytes=16*1024**2,max_stage_evidence_bytes=512*1024**2,
            process_memory_bytes=None,job_memory_bytes=None,max_processes=None,cuda_memory_fraction=None,
            memory_policy='Windows LowMemoryResourceNotification; no free-RAM threshold, allocator quota or hardware batch cap'),
        forensic=dict(B4='OPEN_ROOT_CAUSE_UNRESOLVED',CPU65536='OPEN',automatic_reproducer=False,
            sections=[dict(id='B4.reproducer',authorization='NOT_AUTHORIZED',reason='separate source-pinned forensic manifest and explicit controlled approval required'),
                dict(id='CPU65536.reproducer',authorization='NOT_AUTHORIZED',reason='high-memory crash experiment excluded from ordinary suite')]),
        approval_history=[],automatic_retry=False)
    publish(out/'manifest.json',manifest,new=True)
    (out/'manifest.sha256').write_text(sha(out/'manifest.json')+'\n',encoding='ascii')
    verify(out/'manifest.json',expected=sha(out/'manifest.json'),live=True)
    return manifest

def verify(path,*,expected=None,live=True,assets=True):
    path=Path(path).resolve()
    if expected is None:expected=path.with_suffix('.sha256').read_text(encoding='ascii').strip()
    if not isinstance(expected,str) or len(expected)!=64 or sha(path)!=expected:raise ValueError('manifest SHA256 mismatch')
    m=read(path)
    if m.get('format')!='hyso.full_acceptance.manifest' or type(m.get('schema_version')) is not int or m['schema_version']!=SCHEMA:
        raise ValueError('unsupported manifest')
    if Path(m['output']).resolve()!=path.parent:raise ValueError('manifest output binding mismatch')
    files=m['source_snapshot']['files'];snapshot=Path(m['source_snapshot']['root']).resolve()
    if not snapshot.is_relative_to(path.parent) or digest(files)!=m['source_snapshot']['tree_sha256']:raise ValueError('snapshot identity mismatch')
    actual={p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*') if p.is_file()}
    if actual!=set(files):raise ValueError('unexpected or missing snapshot file; cached/unpinned code prohibited')
    if live and {p.relative_to(Path(m['repository'])).as_posix() for p in source_files(m['repository'])}!=set(files):
        raise ValueError('live source file set changed; prepare a new manifest')
    for rel,expected_hash in files.items():
        p=(snapshot/rel).resolve()
        if not p.is_relative_to(snapshot) or p.is_symlink() or sha(p)!=expected_hash:raise ValueError('snapshot source changed: '+rel)
        if live and sha(Path(m['repository'])/rel)!=expected_hash:raise ValueError('live source drift; prepare a new manifest: '+rel)
    if assets:
        refs=[*m['assets']['models'].values(),m['assets']['dataset'],m['assets']['split_indices'],m['assets']['split_manifest'],
            m['runtime']['python'],*m['runtime']['runtime_files'],*m['runtime'].get('optional_runtime_files',[]),*m['prior_audit'].values()]
        if m.get('preservation_baseline'):refs.append(m['preservation_baseline'])
        if m.get('external_preservation_baseline'):refs.append(m['external_preservation_baseline'])
        refs.extend(x['file'] for x in m['optional'].values() if 'file' in x)
        if 'ncu_plan' in m['optional']:refs.append(m['optional']['ncu_plan']['configuration']['tool'])
        if 'camera_adapter' in m['optional']:
            camera=m['optional']['camera_adapter']['configuration']
            refs.extend(camera[x] for x in ('factory_source','checkpoint','video') if isinstance(camera.get(x),dict) and camera[x].get('path'))
        for r in refs:
            if sha(r['path'])!=r['sha256']:raise ValueError('asset/runtime changed: '+r['path'])
        for raw,expected_hash in m['replay_sources'].items():
            if sha(raw)!=expected_hash:raise ValueError('historical raw evidence changed: '+raw)
    definitions=__import__('full_acceptance.registry',fromlist=['EXPERIMENTS']).EXPERIMENTS
    expected_ids=[r[0] for r in definitions]
    if [x['id'] for x in m['experiments']]!=expected_ids:raise ValueError('experiment registry mismatch')
    for item,definition in zip(m['experiments'],definitions):
        if (item['stage'],item['handler'],item['depends'],item['requires'])!=(definition[1],definition[2],list(definition[4]),list(definition[5])):
            raise ValueError('sealed experiment routing/dependency mismatch')
        if type(item['authorization']['approved']) is not bool:raise ValueError('explicit boolean authorization required')
        if item['automatic_retry'] is not False:raise ValueError('automatic retry prohibited')
        if Path(item['output']).resolve()!=path.parent/'stages'/item['id']:raise ValueError('experiment output escape')
        if item['command'][0]!=m['runtime']['python']['path']:raise ValueError('interpreter mismatch')
        expected_command=[m['runtime']['python']['path'],'-I','-S','-B',str(snapshot/'tools/run_full_acceptance.py'),
            '--worker',item['id'],'--manifest',str(path),'--stage-output',item['output']]
        if item['command']!=expected_command:raise ValueError('exact command binding mismatch')
    return m

def authorize(parent,output,ids,*,expected,basis):
    m=verify(parent,expected=expected)
    if not basis.strip():raise ValueError('approval text required')
    known={e['id'] for e in m['experiments']}
    if not ids or not set(ids)<=known:raise ValueError('unknown or empty explicit authorization list')
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('authorization creates a new exclusive run directory')
    out.mkdir(parents=True);shutil.copytree(m['source_snapshot']['root'],out/'source')
    for name,key in (('preservation-before.json','preservation_baseline'),('external-preservation-before.json','external_preservation_baseline')):
        if m.get(key):
            shutil.copyfile(m[key]['path'],out/name);m[key]=record(out/name)
    m['preservation_excluded_directories']=list(dict.fromkeys([*m.get('preservation_excluded_directories',[]),str(out),m['output']]))
    old_sha=sha(parent);m['run_id']=str(uuid.uuid4());m['output']=str(out);m['source_snapshot']['root']=str(out/'source')
    m['prepared_at']=now();m['approval_history'].append(dict(parent_manifest=record(parent),approved_ids=sorted(set(ids)),basis=basis,at=now()))
    defaults={r['id']:r['authorization'] for r in experiments()}
    for item in m['experiments']:
        # A child manifest must name every desired native approval again. Merely
        # authorizing a different stage cannot silently retry an old GPU stage.
        item['authorization']=dict(defaults[item['id']])
        item['output']=str(out/'stages'/item['id'])
        item['command']=[m['runtime']['python']['path'],'-I','-S','-B',str(out/'source/tools/run_full_acceptance.py'),
            '--worker',item['id'],'--manifest',str(out/'manifest.json'),'--stage-output',item['output']]
        if item['id'] in ids:item['authorization']=dict(approved=True,basis=basis,approved_at=now())
    publish(out/'manifest.json',m,new=True);(out/'manifest.sha256').write_text(sha(out/'manifest.json')+'\n',encoding='ascii')
    verify(out/'manifest.json',expected=sha(out/'manifest.json'))
    return m
