"""P31 bounded owned-process execution and reopenable, hash-chained health journal."""
from dataclasses import dataclass, asdict, field
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import uuid

from .runtime_result import RuntimeResult, _result, _hash, _int, _number, _name, _digest
from hyso.benchmark.result import _write_export


@dataclass(frozen=True)
class WorkloadRecipe:
    factory: str
    configuration: dict = field(default_factory=dict)
    search_path: str | None = None
    gpu_uuid: str | None = None

    def __post_init__(self):
        if type(self.factory) is not str or not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*',self.factory):
            raise ValueError('importable module:factory required')
        if type(self.configuration) is not dict or len(json.dumps(self.configuration,allow_nan=False))>262144:
            raise ValueError('bounded JSON factory configuration required')
        if self.search_path is not None and not Path(self.search_path).is_absolute():
            raise ValueError('absolute factory search directory required')
        if self.gpu_uuid is not None and not re.fullmatch(r'GPU-[a-fA-F0-9-]{16,64}',self.gpu_uuid):
            raise ValueError('explicit physical GPU UUID required')


@dataclass(frozen=True)
class StabilityProtocol:
    """Owned run/evidence bounds; memory and process-count quotas are opt-in.

    OS memory limits constrain private commit, not RSS. Ownership and deadlines
    remain active when quotas are None; no free-memory admission gate is used.
    """
    duration_seconds: float = 1.
    interval_seconds: float = .1
    idle_seconds: float = 0.
    startup_timeout_seconds: float = 45.
    step_timeout_seconds: float = 5.
    max_samples: int = 10000
    chunk_rows: int = 64
    max_journal_bytes: int = 128*1024**2
    seed: int = 0
    process_memory_bytes: int | None = None
    job_memory_bytes: int | None = None
    max_processes: int | None = None

    def __post_init__(self):
        for x in (self.duration_seconds,self.interval_seconds,self.startup_timeout_seconds,self.step_timeout_seconds):_number(x,1e-3)
        _number(self.idle_seconds)
        if (self.duration_seconds>86400 or self.interval_seconds>self.duration_seconds or self.idle_seconds>60
                or self.startup_timeout_seconds>120 or self.step_timeout_seconds>60
                or self.step_timeout_seconds<3*self.interval_seconds):raise ValueError('unsupported duration/heartbeat budget')
        _int(self.max_samples,3,10000);_int(self.chunk_rows,1,512)
        _int(self.max_journal_bytes,65536,256*1024**2);_int(self.seed,0,2**32-1)
        for limit in (self.process_memory_bytes,self.job_memory_bytes):
            if limit is not None:_int(limit,1)
        if self.max_processes is not None:_int(self.max_processes,1,2**32-1)
        if (self.process_memory_bytes is not None and self.job_memory_bytes is not None
                and self.process_memory_bytes>self.job_memory_bytes):raise ValueError('job memory must cover one process')
        if math.ceil((self.duration_seconds+self.idle_seconds)/self.interval_seconds)+4>self.max_samples:
            raise ValueError('requested duration exceeds sample budget; select a larger interval')


def _atomic(path, data):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,allow_nan=False),encoding='utf-8')
    deadline=time.monotonic()+.5
    while True:
        try:os.replace(temporary,path);break
        except PermissionError:
            # Windows metadata readers/AV can briefly deny FILE_SHARE_DELETE.
            # Retry is bounded and remains inside the observed wall-time window.
            if time.monotonic()>=deadline:raise
            time.sleep(.005)


def _load(path, limit=128*1024**2):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>limit:raise ValueError('missing/linked/oversized evidence')
    return json.loads(path.read_text(encoding='utf-8'))


def _size_if_present(path):
    try:return path.stat().st_size
    except FileNotFoundError:return 0  # Atomic publication may rename a scanned .tmp.


class _Journal:
    def __init__(self, root, header, chunk_rows, max_samples):
        self.root=Path(root);self.header=header;self.chunk_rows=chunk_rows;self.max_samples=max_samples
        self.pending=[];self.chunks=[];self.count=0;self.previous=_hash(header)
        _write_export(self.root/'header.json',json.dumps(header,allow_nan=False),False)

    def append(self, row):
        if self.count>=self.max_samples:raise ValueError('journal sample budget exceeded')
        row=dict(row,index=self.count);self.pending.append(row);self.count+=1
        if len(self.pending)>=self.chunk_rows:self.flush()

    def flush(self):
        if not self.pending:return
        index=len(self.chunks)
        chunk=dict(schema_version=1,run_id=self.header['run_id'],index=index,previous_sha256=self.previous,rows=self.pending)
        digest=_hash(chunk);name=f'chunk-{index:06d}.json'
        _write_export(self.root/name,json.dumps(chunk,allow_nan=False),False)
        self.chunks.append(dict(name=name,sha256=digest,rows=len(self.pending)))
        self.previous=digest;self.pending=[]


def _run_owned(root, mode, request, *, startup_timeout, heartbeat_timeout, max_runtime, max_bytes):
    """Windows admission before package/factory imports; kill-on-close Job tree."""
    from ._training_process import _Job
    if os.name!='nt':raise NotImplementedError('owned runtime worker native adapter is Windows-only')
    root=Path(root).absolute();root.mkdir(parents=False,exist_ok=False)
    token=str(uuid.uuid4())
    config=dict(request,mode=mode,token=token,package_root=str(Path(__file__).resolve().parents[2]),
                runtime_paths=[p for p in sys.path if Path(p).is_dir() and Path(p).name in ('site-packages','dist-packages')])
    if not config['runtime_paths']:raise RuntimeError('cannot reproduce runtime site paths')
    (root/'request.json').write_text(json.dumps(config,allow_nan=False),encoding='utf-8')
    # P31/P35 quotas are opt-in; unrelated runtime adapters retain their policy.
    limits=((request['protocol'].get('process_memory_bytes'),request['protocol'].get('job_memory_bytes'))
            if mode in ('stability','providers') else (4*1024**3,6*1024**3))
    process_limit=request['protocol'].get('max_processes') if mode in ('stability','providers') else 4
    job=_Job(*limits,process_limit);process=None;cleanup=None;reason=None;interrupted=None
    started=time.monotonic();ready_seen=None;last_beat=None;last_mtime=None
    stdout=root/'stdout.txt';stderr=root/'stderr.txt'
    try:
        with stdout.open('wb') as out,stderr.open('wb') as err:
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
            gpu=request.get('recipe',{}).get('gpu_uuid')
            if gpu:env['CUDA_VISIBLE_DEVICES']=gpu
            process=subprocess.Popen([getattr(sys,'_base_executable',sys.executable),'-I','-S','-B',
                str(Path(__file__).with_name('_runtime_worker.py')),str(root)],stdout=out,stderr=err,
                env=env,creationflags=subprocess.CREATE_NO_WINDOW)
            job.assign(process)
            _atomic(root/'admit.json',dict(token=token,pid=process.pid))
            while process.poll() is None:
                now=time.monotonic()
                if (root/'ready.json').is_file() and ready_seen is None:ready_seen=last_beat=now
                beat=root/'heartbeat.json'
                if beat.is_file():
                    stamp=beat.stat().st_mtime_ns
                    if stamp!=last_mtime:last_beat=now;last_mtime=stamp
                if ready_seen is None and now-started>startup_timeout:reason='startup_timeout';break
                if ready_seen is not None and now-ready_seen>max_runtime:reason='duration_timeout';break
                if last_beat is not None and now-last_beat>heartbeat_timeout:reason='heartbeat_timeout';break
                files=[p for p in root.iterdir() if p.is_file()]
                if (sum(_size_if_present(p) for p in files)>max_bytes
                        or stdout.stat().st_size+stderr.stat().st_size>65536):reason='resource_limit';break
                time.sleep(.02)
    except BaseException as error:
        if mode!='stability':raise
        reason='controller_interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'controller_error'
        interrupted=error
    finally:
        try:
            job.terminate()
            if process is not None:
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
            deadline=time.monotonic()+5
            while True:
                cleanup=job.observation()
                if cleanup['active_processes']==0 or time.monotonic()>=deadline:break
                time.sleep(.01)
        finally:job.close()
    if cleanup is None or cleanup['active_processes']!=0:raise RuntimeError('owned process tree cleanup failed')
    if interrupted is not None and process is None:raise interrupted
    controller=dict(token=token,launcher_pid=process.pid,exit_code=process.returncode,reason=reason,
        elapsed_seconds=time.monotonic()-started,cleanup=cleanup,kill_on_close=True,
        stderr_tail=stderr.read_bytes()[-4096:].decode('utf-8',errors='replace'))
    if mode in ('stability','providers'):controller['memory_limits']=dict(process_memory_bytes=limits[0],job_memory_bytes=limits[1])
    if mode in ('stability','providers'):controller['process_limit']=process_limit
    _atomic(root/'controller.json',controller)
    if interrupted is not None:raise interrupted
    return root


def measure_stability(recipe, directory, *, protocol=StabilityProtocol()):
    if type(recipe) is not WorkloadRecipe or type(protocol) is not StabilityProtocol:raise TypeError('typed recipe/protocol required')
    if recipe.search_path is not None and not Path(recipe.search_path).is_dir():raise ValueError('missing factory search directory')
    p=protocol
    root=_run_owned(directory,'stability',dict(recipe=asdict(recipe),protocol=asdict(p)),
        startup_timeout=p.startup_timeout_seconds,heartbeat_timeout=p.step_timeout_seconds,
        max_runtime=p.duration_seconds+p.idle_seconds+p.step_timeout_seconds+5,max_bytes=p.max_journal_bytes)
    return reopen_stability(root)


def _read_journal(root):
    root=Path(root);request=_load(root/'request.json',1024**2);controller=_load(root/'controller.json',1024**2)
    if request['mode']=='stability':
        p=StabilityProtocol(**request['protocol']);row_limit=p.max_samples;byte_limit=p.max_journal_bytes
    else:
        from .streaming import StreamProtocol
        p=StreamProtocol(**request['protocol']);row_limit=p.max_frames*8+16;byte_limit=128*1024**2
    header=_load(root/'header.json') if (root/'header.json').exists() else None
    terminal=_load(root/'terminal.json') if (root/'terminal.json').exists() else None
    rows=[];chunks=[];previous=_hash(header) if header else None
    total_bytes=0
    for index,path in enumerate(sorted(root.glob('chunk-*.json'))):
        total_bytes+=path.stat().st_size
        if total_bytes>byte_limit or index>=row_limit:raise ValueError('journal byte/chunk budget exceeded')
        if path.name!=f'chunk-{index:06d}.json':raise ValueError('missing/reordered journal chunk')
        chunk=_load(path)
        if (chunk['index']!=index or chunk['previous_sha256']!=previous or chunk['run_id']!=header['run_id']
                or chunk['schema_version']!=1):raise ValueError('broken journal chain')
        previous=_hash(chunk);rows.extend(chunk['rows'])
        if len(rows)>row_limit:raise ValueError('journal row budget exceeded')
        chunks.append(dict(name=path.name,sha256=previous,rows=len(chunk['rows'])))
    if terminal and (terminal['last_chunk_sha256']!=previous or terminal['chunks']!=chunks):
        raise ValueError('terminal journal manifest mismatch')
    evidence=dict(request={k:request[k] for k in ('token','recipe','protocol','source') if k in request},header=header,rows=rows,
                chunks=chunks,terminal=terminal,controller=controller,
                incomplete_tail=any(root.glob('*.tmp')))
    if request['mode']=='stability' and (root/'worker_failure.json').exists():
        evidence['worker_failure']=_load(root/'worker_failure.json',16384)
    return evidence


def reopen_stability(directory):
    """Committed chunks survive failure. Reopen validates hashes/order/terminal."""
    try:return _result('P31',_read_journal(directory))
    except (KeyError,TypeError,IndexError) as exc:raise ValueError('malformed stability journal') from exc


def _analyze_stability(e):
    p=StabilityProtocol(**e['request']['protocol']);WorkloadRecipe(**e['request']['recipe'])
    c=e['controller'];_int(c['launcher_pid'],1)
    _int(c['cleanup']['active_processes'])
    if type(c['exit_code']) is not int:raise ValueError('invalid worker exit code')
    if c['token']!=e['request']['token'] or c['cleanup']['active_processes']!=0 or c['kill_on_close'] is not True:
        raise ValueError('controller identity/cleanup mismatch')
    if 'memory_limits' in c and c['memory_limits']!=dict(process_memory_bytes=p.process_memory_bytes,job_memory_bytes=p.job_memory_bytes):
        raise ValueError('controller resource policy mismatch')
    if 'process_limit' in c and c['process_limit']!=p.max_processes:raise ValueError('controller process policy mismatch')
    rows=e['rows'];header=e['header'];terminal=e['terminal']
    worker_failure=e.get('worker_failure')
    if worker_failure is not None and (worker_failure['token']!=c['token'] or worker_failure['pid']!=c['launcher_pid']
            or worker_failure['kind'] not in ('oom','device_error','workload_exception','measurement_error')):
        raise ValueError('worker failure identity/category mismatch')
    if len(rows)>p.max_samples:raise ValueError('excess journal rows')
    if header is None:
        if rows or e['chunks'] or terminal:raise ValueError('rows without header')
        return dict(status='aborted',reason=c['reason'] or (worker_failure['kind'] if worker_failure else 'worker_setup_failed'),completed=0,
                    steps=0,completed_per_second=None,duration_seconds=None,
                    requested_duration_seconds=p.duration_seconds,coverage='none')
    if header['run_id']!=e['request']['token'] or header['pid']!=c['launcher_pid']:
        raise ValueError('journal worker/run association mismatch')
    if e['request']['recipe'].get('gpu_uuid') is not None and header['workload']['device_id']!=e['request']['recipe']['gpu_uuid']:
        raise ValueError('workload physical GPU association mismatch')
    if _hash(header['protocol'])!=_hash(e['request']['protocol']):raise ValueError('journal protocol mismatch')
    if header['source']['factory']!=e['request']['recipe']['factory'] or header['source']['configuration']!=e['request']['recipe']['configuration']:
        raise ValueError('workload source association mismatch')
    _name(header['clock_id']);_name(header['workload']['device_id']);_name(header['workload']['unit'])
    previous=_hash(header);cursor=0
    for i,chunk in enumerate(e['chunks']):
        n=chunk['rows'];_int(n,1,p.chunk_rows)
        data=dict(schema_version=1,run_id=header['run_id'],index=i,previous_sha256=previous,rows=rows[cursor:cursor+n])
        previous=_hash(data);cursor+=n
        if chunk['name']!=f'chunk-{i:06d}.json' or chunk['sha256']!=previous:raise ValueError('journal content/hash mismatch')
    if cursor!=len(rows):raise ValueError('journal row count mismatch')
    last_ns=last_completed=last_steps=last_work=0;gaps=[];failures=[];temperatures=[];memory=[];cpu=[];performance=[]
    for i,row in enumerate(rows):
        if row['index']!=i:raise ValueError('missing/duplicate sample sequence')
        stamp=row['timestamp_ns'];_int(stamp)
        if stamp<=last_ns:raise ValueError('nonmonotonic timeline')
        if last_ns and stamp-last_ns>2*p.interval_seconds*1e9:gaps.append(dict(before_index=i-1,after_index=i,gap_ns=stamp-last_ns))
        for key,old in (('completed',last_completed),('steps',last_steps),('work_ns',last_work)):
            _int(row[key],old)
        if row['completed']<row['steps']:raise ValueError('fewer completions than successful steps')
        if header['workload']['kind']=='model_inference':
            batch=header['workload']['input_shape'][0];_int(batch,1)
            if row['completed']!=row['steps']*batch:raise ValueError('model completion/image batch accounting mismatch')
        if i==0 and (row['completed'] or row['steps'] or row['work_ns']):raise ValueError('first observation must precede work')
        if last_ns:
            elapsed=(stamp-last_ns)/1e9;steps=row['steps']-last_steps
            performance.append(dict(start_ns=last_ns,end_ns=stamp,completed=row['completed']-last_completed,
                completed_per_second=(row['completed']-last_completed)/elapsed,
                mean_step_work_ms=(row['work_ns']-last_work)/steps/1e6 if steps else None))
        if row['last_completion_ns'] is not None:
            _int(row['last_completion_ns'])
            if row['last_completion_ns']>stamp:raise ValueError('completion after observation')
        if row['phase'] not in ('idle','work','terminal','failure'):raise ValueError('unknown sample phase')
        if row['phase']=='idle' and (row['completed'] or row['steps']):raise ValueError('idle contains work')
        rss=row['rss']
        if rss is not None:
            for key in ('rss_bytes','lifetime_hwm_bytes','timestamp_ns','query_start_ns','query_end_ns'):_int(rss[key])
            if not rss['query_start_ns']<=rss['timestamp_ns']<=rss['query_end_ns']<=stamp or rss['rss_bytes']>rss['lifetime_hwm_bytes']:
                raise ValueError('invalid P10 memory observation')
            memory.append(rss['rss_bytes'])
        counter=row['cpu']
        if counter is not None:
            for key in ('query_start_ns','query_end_ns','timestamp_ns','kernel_ns','user_ns'):_int(counter[key])
            if not counter['query_start_ns']<=counter['timestamp_ns']<=counter['query_end_ns']<=stamp:
                raise ValueError('CPU health observation clock mismatch')
            cpu.append(counter)
        alloc=row['allocator']
        if alloc is not None:
            if alloc['device_id']!=header['workload']['device_id']:raise ValueError('allocator device mismatch')
            _int(alloc['allocated_bytes']);_int(alloc['reserved_bytes'],alloc['allocated_bytes'])
        gpu=row['gpu']
        if gpu.get('backend')=='nvidia-smi.power.draw.instant':
            from .energy import _validate_power_sample
            _validate_power_sample(gpu,header['workload']['device_id'],header['clock_id'],require_power=gpu['status']=='measured')
            if gpu['query_end_ns']>stamp:raise ValueError('sensor after observation')
            if gpu['temperature_c'] is not None:temperatures.append(gpu['temperature_c'])
        if row['failure'] is not None:failures.append(dict(index=i,reason=row['failure']))
        last_ns=stamp;last_completed=row['completed'];last_steps=row['steps'];last_work=row['work_ns']
    elapsed=None;complete=False
    if terminal is not None:
        if terminal['last_chunk_sha256']!=previous or terminal['chunks']!=e['chunks'] or terminal['rows']!=len(rows):
            raise ValueError('terminal ledger mismatch')
        start,end=terminal['work_start_ns'],terminal['work_end_ns'];_int(start);_int(end,start)
        if terminal['completed']!=last_completed or terminal['steps']!=last_steps:raise ValueError('terminal completion mismatch')
        if any(r['last_completion_ns'] is not None and not start<=r['last_completion_ns']<=end for r in rows):
            raise ValueError('completion outside workload window')
        elapsed=(end-start)/1e9
        complete=(terminal['status']=='completed' and terminal['closed'] is True and not failures and last_steps>0
            and elapsed>=p.duration_seconds and c['reason'] is None and c['exit_code']==0 and not e['incomplete_tail'])
    from .utilization import _cpu_intervals
    from ._stability_diagnostics import _validate_extensions
    _validate_extensions(rows,terminal,header)
    cpu_intervals=_cpu_intervals(cpu,header['runtime']['logical_cores'],int(p.interval_seconds*1e9))
    return dict(status='completed' if complete else 'aborted',reason=None if complete else c['reason'] or
        ((terminal['reason'] or 'duration_or_completion_coverage_not_met') if terminal else 'missing_terminal'),completed=last_completed,steps=last_steps,
        duration_seconds=elapsed,requested_duration_seconds=p.duration_seconds,
        coverage='requested_duration_observed' if complete else 'committed_prefix_only',
        completed_per_second=last_completed/elapsed if elapsed and complete else None,
        gaps=gaps,gap_count=len(gaps),failures=failures,performance_timeline=performance,cpu_timeline=cpu_intervals,
        health=dict(rss_samples=len(memory),rss_min_bytes=min(memory) if memory else None,
            rss_max_bytes=max(memory) if memory else None,thermal_samples=len(temperatures),
            max_temperature_c=max(temperatures) if temperatures else None,
            thermal_status='measured' if temperatures else 'unsupported_or_not_requested',cpu_samples=len(cpu)),
        clock_scope='one_owned_process_perf_counter_ns',includes_health_sampling_overhead=True,
        long_duration_claim='only_requested_and_observed_seconds_no_hours_extrapolation')


class InferenceWorkload:
    """Trusted eager helper with synchronous completions and validation overhead.

    cuda_memory_fraction=None leaves the allocator quota unset. An explicit
    fraction constrains only the admitted worker allocator, not physical VRAM.
    """
    def __init__(self, model, inputs, *, device='cpu', gpu_uuid=None, cuda_memory_fraction=None):
        import copy
        import torch
        from hyso.benchmark.runner import _profile,_descriptor,_Snapshot,_output
        self.torch=torch;self.device=torch.device(device)
        if cuda_memory_fraction is not None:
            _number(cuda_memory_fraction,1e-9)
            if cuda_memory_fraction>1 or self.device.type!='cuda':raise ValueError('explicit CUDA allocator fraction in (0,1] required')
        if self.device.type not in ('cpu','cuda'):raise NotImplementedError('CPU/CUDA only')
        if self.device.type=='cuda' and os.environ.get('HYSO_RUNTIME_ADMITTED_PID')!=str(os.getpid()):
            raise ValueError('CUDA helper requires an admitted owned runtime worker')
        if self.device.type=='cuda' and (gpu_uuid is None or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu_uuid or self.device.index not in (None,0)):
            raise ValueError('CUDA worker requires one explicit UUID-visible device')
        if self.device.type=='cuda':
            self.device=torch.device('cuda',0)
            if cuda_memory_fraction is not None:
                torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction,self.device)
        self.model=copy.deepcopy(model).to(self.device).eval();self.inputs=inputs.detach().clone().to(self.device).contiguous()
        modules,entries=_profile(self.model,self.inputs.dtype,self.device)
        descriptor,digest=_descriptor(modules,entries)
        self.guard=_Snapshot(modules,entries,self.inputs);self.validate_output=_output
        self.identity=dict(kind='model_inference',unit='image',device_id=gpu_uuid if self.device.type=='cuda' else 'cpu',
            model_sha256=digest,structural_descriptor=descriptor,input_shape=list(self.inputs.shape),backend='pytorch_eager',
            cuda_memory_fraction=cuda_memory_fraction)

    def step(self):
        with self.torch.inference_mode():
            output=self.model(self.inputs)
            if self.device.type=='cuda':self.torch.cuda.synchronize(self.device)
            self.validate_output(output,len(self.inputs),device=self.device)
            self.guard.check(full=True)
            return len(self.inputs)

    def close(self):
        self.guard.check(full=True)


def _factory(recipe):
    if recipe.get('search_path'):sys.path.append(recipe['search_path'])
    module,qualified=recipe['factory'].split(':');loaded=importlib.import_module(module);factory=loaded
    for part in qualified.split('.'):factory=getattr(factory,part)
    path=inspect.getsourcefile(loaded)
    source=dict(factory=recipe['factory'],configuration=recipe['configuration'],
                module_path=path,module_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None)
    return factory(**recipe['configuration']),source


def _worker_stability(root, request):
    from .memory import _observe
    from . import _telemetry
    import torch
    import numpy as np
    import random
    p=StabilityProtocol(**request['protocol']);seed=p.seed
    random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed);torch.set_num_threads(1)
    workload,source=_factory(request['recipe']);identity=workload.identity
    _name(identity['device_id']);_name(identity['unit'])
    header=dict(run_id=request['token'],pid=os.getpid(),clock_id=request['token']+':perf_counter_ns',
        workload=identity,source=source,protocol=asdict(p),runtime=dict(python=platform.python_version(),
        torch=str(torch.__version__),platform=platform.platform(),logical_cores=os.cpu_count() or 1))
    journal=_Journal(root,header,p.chunk_rows,p.max_samples)
    _atomic(root/'ready.json',dict(pid=os.getpid(),token=request['token']))
    completed=steps=work_ns=attempted=0;last_completion=None;failure=None;closed=False;start=end=0
    mean_ns=m2_ns2=0.;minimum=maximum=None;failure_kind=None;stage='telemetry'
    def sample(phase, error=None):
        try:rss=_observe()
        except (OSError,RuntimeError):rss=None
        try:cpu=_telemetry.cpu_observe()
        except (OSError,NotImplementedError):cpu=None
        allocator=None
        if identity['device_id']!='cpu' and torch.cuda.is_initialized():
            query_start=time.perf_counter_ns()
            allocator=dict(device_id=identity['device_id'],allocated_bytes=torch.cuda.memory_allocated(0),reserved_bytes=torch.cuda.memory_reserved(0),
                           query_start_ns=query_start,query_end_ns=time.perf_counter_ns())
        from .energy import observe_nvidia_power
        gpu=observe_nvidia_power(identity['device_id'],clock_id=header['clock_id'],timeout_seconds=1.) if identity['device_id']!='cpu' else dict(status='not_requested')
        gpu_utilization=(_telemetry.gpu_observe(identity['device_id'],1.) if identity['device_id']!='cpu'
                         else dict(status='not_requested'))
        tracker=getattr(workload,'retention_tracker',None)
        retention=tracker.snapshot() if tracker is not None else None
        journal.append(dict(timestamp_ns=time.perf_counter_ns(),phase=phase,completed=completed,steps=steps,work_ns=work_ns,
            last_completion_ns=last_completion,rss=rss,cpu=cpu,allocator=allocator,gpu=gpu,retention=retention,failure=error,
            attempted_steps=attempted,failure_kind=failure_kind,
            step_timing=dict(count=steps,mean_ns=mean_ns if steps else None,m2_ns2=m2_ns2,min_ns=minimum,max_ns=maximum),
            gpu_utilization=gpu_utilization))
        _atomic(root/'heartbeat.json',dict(sequence=journal.count))
    try:
        idle_deadline=time.perf_counter()+p.idle_seconds
        while time.perf_counter()<idle_deadline:
            sample('idle');time.sleep(min(p.interval_seconds,max(0,idle_deadline-time.perf_counter())))
        start=time.perf_counter_ns();due=start
        sample('work');due+=int(p.interval_seconds*1e9)
        while time.perf_counter_ns()-start<p.duration_seconds*1e9:
            stage='step';attempted+=1
            a=time.perf_counter_ns();count=workload.step();b=time.perf_counter_ns()
            _int(count,1);completed+=count;steps+=1;work_ns+=b-a;last_completion=b
            duration=b-a;delta=duration-mean_ns;mean_ns+=delta/steps;m2_ns2+=delta*(duration-mean_ns)
            minimum=duration if minimum is None else min(minimum,duration)
            maximum=duration if maximum is None else max(maximum,duration)
            stage='telemetry'
            if b>=due:
                sample('work');due=time.perf_counter_ns()+int(p.interval_seconds*1e9)
        end=time.perf_counter_ns();sample('terminal')
    except Exception as exc:
        failure=type(exc).__name__+': '+str(exc)[:1024];end=time.perf_counter_ns()
        kind=_failure_kind(exc)
        failure_kind=kind if stage=='step' or kind in ('oom','device_error') else 'measurement_error'
        if start==0:start=end
        try:sample('failure',failure)
        except Exception:pass
    finally:
        try:workload.close();closed=True
        except Exception as exc:
            failure=failure or 'close: '+str(exc)[:1024];failure_kind=failure_kind or _failure_kind(exc)
        journal.flush()
        terminal=dict(status='completed' if failure is None else 'failed',reason=failure,closed=closed,
            work_start_ns=start,work_end_ns=end,completed=completed,steps=steps,rows=journal.count,
            chunks=journal.chunks,last_chunk_sha256=journal.previous,attempted_steps=attempted,failure_kind=failure_kind)
        _atomic(root/'terminal.json',terminal)


def _failure_kind(error):
    """Classify actual exceptions; device occupancy never establishes a cause."""
    import torch
    if isinstance(error,(MemoryError,torch.cuda.OutOfMemoryError)) or (
            isinstance(error,RuntimeError) and 'DefaultCPUAllocator: not enough memory' in str(error)):
        return 'oom'
    if isinstance(error,RuntimeError) and any(marker in str(error) for marker in
            ('CUDA error:', 'CUDA driver error:', 'device-side assert', 'device lost')):
        return 'device_error'
    return 'workload_exception'
