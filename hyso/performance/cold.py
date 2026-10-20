"""Fresh-interpreter cold trials, plain recipes and explicitly owned worker cleanup."""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time

import torch

from .inference import ThroughputProtocol, _base
from .result import ColdStartResult, _plain, _positive, _integer

PHASES=('module_import','model_construction','artifact_load','runtime_prepare','first_inference','warmup','warm_reference')


@dataclass(frozen=True,init=False)
class ColdStartRecipe:
    __slots__=('_payload',)
    _payload: str

    def __init__(self,factory,*,model_kwargs=None,input_shape=(1,1,5,7),device='cpu',precision='fp32',
                 artifact_path=None,artifact_sha256=None,factory_search_path=None,seed=0):
        import re
        if type(factory) is not str or re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*',factory) is None:
            raise ValueError('factory must be importable module:qualified_name')
        ThroughputProtocol(device=device,precision=precision,batch_count=1)
        from hyso.benchmark.runner import _shape
        shape=_shape(input_shape); _integer(seed)
        if artifact_path is not None and (type(artifact_sha256) is not str or re.fullmatch('[0-9a-fA-F]{64}',artifact_sha256) is None):
            raise ValueError('artifact needs expected SHA256 and explicit weights_only load policy')
        if artifact_path is None and artifact_sha256 is not None: raise ValueError('SHA requires artifact_path')
        kwargs=_plain({} if model_kwargs is None else model_kwargs)
        if type(kwargs) is not dict: raise TypeError('model_kwargs must be plain dict')
        data=dict(factory=factory,model_kwargs=kwargs,input_shape=list(shape),device=device,precision=precision,
            artifact_path=str(Path(artifact_path).resolve()) if artifact_path is not None else None,
            artifact_sha256=artifact_sha256.lower() if artifact_sha256 is not None else None,
            factory_search_path=str(Path(factory_search_path).resolve()) if factory_search_path is not None else None,seed=seed)
        object.__setattr__(self,'_payload',json.dumps(data,allow_nan=False))

    def to_dict(self): return json.loads(self._payload)


@dataclass(frozen=True)
class ColdStartProtocol:
    trials: int = 3
    timeout_seconds: float = 60.
    warmup_iterations: int = 2
    warm_samples: int = 3

    def __post_init__(self):
        _integer(self.trials,1); _positive(self.timeout_seconds)
        _integer(self.warmup_iterations); _integer(self.warm_samples,1)


def _missing(reason,status='not_observed'):
    return dict(start_ns=None,end_ns=None,elapsed_ms=None,unit='ms',clock='worker_perf_counter_ns',
                status=status,reason=reason,device=None,backend=None,included_operations=[])


def _terminate(worker):
    if worker.poll() is not None: return
    if os.name=='nt':
        # Windows venv redirectors may own a different real Python PID. Kill
        # this still-live launcher and its owned tree, not just the redirector.
        executable=Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/taskkill.exe'
        killed=subprocess.run([str(executable),'/PID',str(worker.pid),'/T','/F'],capture_output=True,
                              creationflags=subprocess.CREATE_NO_WINDOW)
        if killed.returncode and worker.poll() is None:
            raise RuntimeError('failed to terminate owned cold-worker process tree')
    else:
        worker.kill()


def _trial(recipe,plan,index,root):
    request=root/('recipe-'+str(index)+'.json'); output=root/('trial-'+str(index)+'.json')
    request.write_text(json.dumps(dict(recipe=recipe,protocol=asdict(plan)),allow_nan=False),encoding='utf-8')
    package_root=Path(__file__).resolve().parents[2]
    command=[sys.executable,'-I','-B',str(Path(__file__).with_name('_cold_worker.py')),str(request),str(output),str(package_root)]
    ready=[]; worker=None; reader=None; status=None; timed_out=False
    start=time.perf_counter_ns()
    # stderr is bounded on readout, not kept in RAM while a trusted factory runs.
    with tempfile.TemporaryFile(mode='w+b') as stderr:
        try:
            worker=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=stderr,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            def read_ready():
                for line in iter(lambda:worker.stdout.readline(65537),b''):
                    if line.startswith(b'@@HYSO_READY ') and not ready:
                        try: ready.append((time.perf_counter_ns(),json.loads(line[13:])))
                        except (ValueError,UnicodeDecodeError): pass
            reader=threading.Thread(target=read_ready,name='hyso-cold-ready',daemon=True); reader.start()
            try: worker.wait(timeout=plan.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out=True; _terminate(worker); worker.wait()
            reader.join()
            end=time.perf_counter_ns()
            status='timed_out' if timed_out else 'crashed' if worker.returncode not in (0,1) else 'failed'
            if not timed_out and output.is_file():
                body=_plain(json.loads(output.read_text(encoding='utf-8')))
                if not ready or body['worker_pid']!=ready[0][1]['pid']: raise ValueError('worker ready/output identity mismatch')
                status=body['status']
            else:
                body=dict(worker_pid=ready[0][1]['pid'] if ready else None,phases={name:_missing(status) for name in PHASES},
                    provenance=dict(device=recipe['device'],worker_source=str(Path(__file__).with_name('_cold_worker.py'))),
                    first_output=None,warm_samples_ms=[],artifact=None,error=status)
            receipt=ready[0][0] if ready else None
            startup=dict(start_ns=start,end_ns=receipt,elapsed_ms=(receipt-start)/1e6 if receipt is not None else None,
                unit='ms',clock='parent_perf_counter_ns',status='measured' if receipt is not None else 'not_observed',
                reason=None if receipt is not None else status,device='host',backend='python_subprocess',
                included_operations=['launcher_and_worker_startup','bootstrap','ready_pipe_receipt_and_reader_scheduling'])
            stderr.seek(0,2); size=stderr.tell(); stderr.seek(max(0,size-4096))
            body.update(trial_index=index,status=status,parent_start_ns=start,parent_end_ns=end,
                parent_elapsed_ms=(end-start)/1e6,exit_code=worker.returncode,process_terminated=worker.poll() is not None,
                launcher_pid=worker.pid,termination_policy='windows_owned_process_tree' if os.name=='nt' else 'owned_process_kill',
                startup_to_ready_receipt=startup,stderr_tail=stderr.read().decode('utf-8',errors='replace'))
            return body
        finally:
            if worker is not None:
                if worker.poll() is None: _terminate(worker)
                worker.wait()
                if reader is not None and reader.ident is not None: reader.join()
                if worker.stdout is not None: worker.stdout.close()


def measure_cold_start(recipe,*,protocol=ColdStartProtocol()):
    """New interpreter per trial. No live model/Tensor/handle crosses the boundary.

    Trusted factory code is importable by name; artifact loading always verifies
    SHA then uses weights_only=True. OS page-cache coldness is unknown, never reset.
    Failures/timeouts/crashes are terminal trial evidence, not successful latency.
    """
    if type(recipe) is not ColdStartRecipe or type(protocol) is not ColdStartProtocol:
        raise TypeError('expected ColdStartRecipe and ColdStartProtocol')
    config=recipe.to_dict()
    with tempfile.TemporaryDirectory(prefix='hyso-cold-') as temporary:
        rows=[_trial(config,protocol,index,Path(temporary)) for index in range(protocol.trials)]
    data=_base('cold_start','P05',dict(id='fresh_interpreter_recipe',version=1,scope='fresh_worker_phases',
        configuration=asdict(protocol),recipe=config,start_method='fresh_subprocess_exec',page_cache='unknown_not_flushed',
        startup_scope='parent_launch_to_ready_receipt_including_pipe_and_reader_scheduling',
        parent_total_scope='launch_through_worker_exit_including_result_publication_and_shutdown',
        load_policy='sha256_then_torch_load_weights_only_cpu',clock_domains=['parent_perf_counter_ns','worker_perf_counter_ns']),
        dict(model=dict(factory=config['factory'],model_kwargs=config['model_kwargs']),input=dict(shape=config['input_shape'],seed=config['seed']),
             environment=dict(device=config['device'],python=platform.python_version(),torch=str(torch.__version__),os=platform.platform()),
             artifact=dict(path=config['artifact_path'],expected_sha256=config['artifact_sha256'],
                           format='pytorch.state_dict' if config['artifact_path'] is not None else None)))
    counts={status:sum(row['status']==status for row in rows) for status in ('measured','failed','unsupported','timed_out','crashed')}
    data.update(requested=dict(trials=protocol.trials,warm_samples=protocol.warm_samples),actual=dict(trials=len(rows),**counts),
        measurements=_cold_metrics(rows),raw_evidence=dict(trials=rows))
    data['diagnostics']['limitations']=['Fresh interpreter, not guaranteed cold OS page cache, disk/controller cache or GPU driver state.',
        'Parent startup is ready-message receipt with pipe/thread overhead; worker phase clocks are not subtracted from parent timestamps.',
        'Model construction is CPU-only; factories must not pre-initialize CUDA or hide model loading.',
        'CUDA preparation includes context init, model/input movement and drain; first inference and warm reference are separate.',
        'Factory code is trusted caller code, not a sandbox; no conversion/export backend is implied.']
    return ColdStartResult(data)


def _cold_metrics(rows):
    values=[row['phases']['first_inference']['elapsed_ms'] for row in rows if row['status']=='measured']
    import math
    return dict(first_inference_mean_ms=dict(value=math.fsum(values)/len(values) if values else None,unit='ms',
        status='derived' if values else 'unavailable',reason=None if values else 'no_successful_trials'))


def _validate_cold(data):
    protocol=data['protocol']; p=ColdStartProtocol(**protocol['configuration']); recipe=ColdStartRecipe(**protocol['recipe'])
    if protocol['id']!='fresh_interpreter_recipe' or protocol['scope']!='fresh_worker_phases' or protocol['page_cache']!='unknown_not_flushed':
        raise ValueError('invalid cold-start scope/cache contract')
    rows=data['raw_evidence']['trials']
    if len(rows)!=p.trials: raise ValueError('missing cold trials')
    for index,row in enumerate(rows):
        if row['trial_index']!=index or row['status'] not in ('measured','failed','unsupported','timed_out','crashed') or row['process_terminated'] is not True:
            raise ValueError('invalid trial identity/status/process cleanup')
        _integer(row['launcher_pid'],1)
        if row['worker_pid'] is not None: _integer(row['worker_pid'],1)
        elif row['status']=='measured': raise ValueError('successful worker needs observed PID')
        _integer(row['parent_start_ns']); _integer(row['parent_end_ns'])
        if row['parent_end_ns']<=row['parent_start_ns'] or row['parent_elapsed_ms']!=(row['parent_end_ns']-row['parent_start_ns'])/1e6:
            raise ValueError('invalid parent duration')
        previous=0
        if set(row['phases'])!=set(PHASES): raise ValueError('missing cold phase')
        for name in PHASES:
            phase=row['phases'][name]
            if phase['clock']!='worker_perf_counter_ns' or phase['unit']!='ms': raise ValueError('invalid worker clock/unit')
            if phase['status'] in ('measured','failed'):
                _integer(phase['start_ns']); _integer(phase['end_ns'])
                if not previous<=phase['start_ns']<=phase['end_ns'] or phase['elapsed_ms']!=(phase['end_ns']-phase['start_ns'])/1e6:
                    raise ValueError('invalid phase time/order')
                previous=phase['end_ns']
            elif (phase['status'] not in ('disabled','not_observed') or not phase['reason'] or
                  any(phase[k] is not None for k in ('start_ns','end_ns','elapsed_ms'))): raise ValueError('missing phase cannot invent time')
        startup=row['startup_to_ready_receipt']
        if startup['clock']!='parent_perf_counter_ns': raise ValueError('startup wrong clock domain')
        if startup['status']=='measured':
            if not row['parent_start_ns']<=startup['end_ns']<=row['parent_end_ns'] or startup['elapsed_ms']!=(startup['end_ns']-startup['start_ns'])/1e6:
                raise ValueError('invalid startup receipt')
        elif startup['elapsed_ms'] is not None: raise ValueError('missing startup must be null')
        if row['status']=='measured':
            if row['exit_code']!=0 or any(row['phases'][name]['status']!='measured' for name in PHASES if name!='artifact_load'):
                raise ValueError('success without completed phases')
            if len(row['warm_samples_ms'])!=p.warm_samples or any(v<=0 for v in row['warm_samples_ms']): raise ValueError('missing warm reference')
    counts={status:sum(r['status']==status for r in rows) for status in ('measured','failed','unsupported','timed_out','crashed')}
    if data['actual']!=dict(trials=len(rows),**counts) or data['requested']!=dict(trials=p.trials,warm_samples=p.warm_samples):
        raise ValueError('inconsistent cold trial counts')
    if data['measurements']!=_cold_metrics(rows): raise ValueError('cold aggregates disagree with trials')
