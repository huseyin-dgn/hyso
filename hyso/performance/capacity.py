"""P17 finite worker admission; timing/forward probes reuse the P06/P08 core."""
from dataclasses import asdict, dataclass
import errno
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import uuid

from .cold import ColdStartRecipe, _terminate
from .inference import _base
from .result import CapacityResult, PrecisionResult, _integer, _positive, _plain


@dataclass(frozen=True)
class BatchSearchProtocol:
    candidates: tuple[int, ...]
    maximum_batch: int
    cpu_memory_limit_bytes: int
    input_limit_bytes: int
    timeout_seconds: float
    stop_policy: str = 'exhaust_candidates'
    warmup_iterations: int = 2
    measure_iterations: int = 10
    repeats: int = 1
    cuda_memory_fraction: float = .1

    def __post_init__(self):
        if type(self.candidates) not in (tuple,list): raise TypeError('finite list/tuple required')
        points=tuple(self.candidates)
        _integer(self.maximum_batch,1)
        if not points or len(points)>512: raise ValueError('need 1..512 explicit candidates')
        for b in points:
            _integer(b,1)
            if b>self.maximum_batch: raise ValueError('candidate exceeds maximum_batch')
        if len(set(points))!=len(points): raise ValueError('duplicate candidate')
        object.__setattr__(self,'candidates',points)
        _integer(self.cpu_memory_limit_bytes,256*1024**2); _integer(self.input_limit_bytes,1)
        _positive(self.timeout_seconds)
        if self.timeout_seconds>3600: raise ValueError('worker timeout exceeds 3600 seconds')
        if self.stop_policy not in ('exhaust_candidates','first_infeasible'): raise ValueError('unknown stop policy')
        _integer(self.warmup_iterations); _integer(self.measure_iterations,1); _integer(self.repeats,1)
        if (self.warmup_iterations+self.measure_iterations)*self.repeats>10000:
            raise ValueError('forward budget exceeds 10000')
        _positive(self.cuda_memory_fraction)
        if self.cuda_memory_fraction>1: raise ValueError('allocator fraction exceeds one')


def _configuration(plan):
    d=asdict(plan); d['candidates']=list(plan.candidates); return d


def _row(batch,recipe):
    shape=list(recipe['input_shape']);shape[0]=batch
    return dict(candidate_id=str(uuid.uuid4()),batch=batch,shape=shape,
        device=recipe['device'],precision=recipe['precision'],backend='pytorch_eager',
        status='not_run',classification='unknown',reason='prior_stop',actual_batch=None,
        forward_completed=False,latency_mean_ms=None,memory=None,sweep=None,
        worker_pid=None,launcher_pid=None,exit_code=None,process_terminated=True,
        worker_phase=None,runtime_ready=False,execution_seconds=None,stderr_tail='',provenance=None)


def _candidate(recipe,plan,batch,root):
    from ._batch_worker import _admission
    row=_row(batch,recipe)
    admission=_admission(row['shape'],recipe['precision'],recipe['device'],
        plan.input_limit_bytes,plan.cpu_memory_limit_bytes)
    if admission['reason']:
        row.update(status='resource_limit',reason=admission['reason'],
            memory=dict(admission=admission))
        return row  # unknown, no worker/factory/forward; caller stops admission.
    from .cold import _owned_worker_launcher
    launcher = _owned_worker_launcher.get()
    if launcher is not None:
        return launcher('capacity', recipe, plan, batch, root)
    item=root/str(batch);item.mkdir()
    request=item/'request.json';output=item/'result.json';ready=item/'ready.json'
    request.write_text(json.dumps(dict(recipe=recipe,protocol=_configuration(plan),batch=batch)),encoding='utf-8')
    package_root=Path(__file__).resolve().parents[2]
    command=[sys.executable,'-I','-B',str(Path(__file__).with_name('_batch_worker.py')),
             str(request),str(output),str(ready),str(package_root)]
    worker=None;timed_out=False;start=time.perf_counter()
    # Files avoid unbounded in-memory stdout/stderr capture and pipe deadlocks.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            try:
                try:
                    worker=subprocess.Popen(command,stdout=stdout,stderr=stderr,
                        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                except OSError as error:
                    # No process handle was returned. Preserve earlier candidate
                    # evidence without calling this a forward OOM or native crash.
                    winerror=getattr(error,'winerror',None)
                    resource=(winerror in (8,14,1450,1455) if winerror is not None
                              else error.errno==errno.ENOMEM)
                    row.update(status='resource_limit' if resource else 'failed',
                        reason=f'worker launch failed (errno={error.errno}, winerror={winerror}): {error}',
                        execution_seconds=time.perf_counter()-start)
                    row['classification']=_classification(row['status'])
                    return row
                row['launcher_pid']=worker.pid
                try: worker.wait(timeout=plan.timeout_seconds)
                except subprocess.TimeoutExpired: timed_out=True
            finally:
                if worker is not None:
                    cleanup_deadline=time.monotonic()+10.
                    if worker.poll() is None:
                        _terminate(worker,timeout=max(0.,cleanup_deadline-time.monotonic()))
                    worker.wait(timeout=max(0.,cleanup_deadline-time.monotonic()))
                    row['process_terminated']=worker.poll() is not None
            row.update(exit_code=worker.returncode,execution_seconds=time.perf_counter()-start)
            if ready.is_file():
                receipt=json.loads(ready.read_text(encoding='utf-8'))
                row.update(worker_pid=receipt['pid'],worker_phase=receipt['phase'],runtime_ready=receipt['runtime_ready'])
            if timed_out: row.update(status='timed_out',reason='owned worker timeout')
            elif worker.returncode!=0:
                row.update(status='crashed',reason='unexpected worker exit; not classified as OOM')
            elif output.is_file():
                if output.stat().st_size>16*1024**2: raise ValueError('worker result exceeds evidence budget')
                body=_plain(json.loads(output.read_text(encoding='utf-8')))
                if body.pop('worker_pid')!=row['worker_pid']: raise ValueError('worker identity mismatch')
                allowed={'status','reason','actual_batch','forward_completed','latency_mean_ms','memory','sweep','provenance'}
                if set(body)!=allowed: raise ValueError('invalid worker receipt')
                row.update(body)
            else: row.update(status='failed',reason='missing worker result')
        finally:
            stderr.seek(0,2);size=stderr.tell();stderr.seek(max(0,size-4096))
            row['stderr_tail']=stderr.read().decode('utf-8',errors='replace')
    row['classification']=_classification(row['status'])
    return row


def _classification(status):
    return 'feasible' if status=='feasible' else 'infeasible' if status=='infeasible_oom' else 'unknown'


def _summary(rows,plan):
    aborted=any(r['status'] not in ('feasible','infeasible_oom','not_run') for r in rows)
    feasible=[r['batch'] for r in rows if r['status']=='feasible']
    observed=max(feasible,default=None)
    classified=[r['batch'] for r in rows if r['classification']!='unknown']
    exhaustive=len(classified)==plan.maximum_batch and set(classified)==set(range(1,plan.maximum_batch+1))
    complete=all(r['status']!='not_run' for r in rows)
    return dict(status='aborted' if aborted else 'complete' if complete else 'partial',
        reason='unexpected_failure_stopped_admission' if aborted else None if complete else 'stop_policy',
        maximum_observed_feasible_batch=None if aborted else observed,
        prior_observed_feasible_batch=observed,
        maximum_proven_within_range=observed if exhaustive and not aborted else None,
        candidate_list_complete=complete,range_exhaustive=exhaustive,
        evidence_quality='all_integer_points_observed' if exhaustive and not aborted else 'explicit_points_only',
        monotonicity='not_assumed',search_bound=plan.maximum_batch,
        memory_scope='worker_private_commit_budget; process_RSS_snapshots; separate_single_forward_CUDA_allocator_probe; not_physical_VRAM')


def find_max_batch_size(recipe,*,protocol):
    """Search trusted CPU factory recipes in fresh bounded Windows workers.

    No live model, Tensor, stream or CUDA context is transferred. CPU memory is
    hard-limited by a worker Job Object; CUDA allocator fraction applies only in
    that child. An unexpected failure always stops admission and suppresses the
    successful maximum. Explicit OOM may continue in a NEW worker by policy.
    """
    if type(recipe) is not ColdStartRecipe: raise TypeError('expected ColdStartRecipe')
    if type(protocol) is not BatchSearchProtocol: raise TypeError('expected BatchSearchProtocol')
    r=recipe.to_dict();rows=[];stopped=False
    with tempfile.TemporaryDirectory(prefix='hyso-capacity-') as directory:
        for batch in protocol.candidates:
            row=_row(batch,r) if stopped else _candidate(r,protocol,batch,Path(directory))
            rows.append(row)
            if row['status'] not in ('feasible','infeasible_oom'): stopped=True
            if row['status']=='infeasible_oom' and protocol.stop_policy=='first_infeasible': stopped=True
    data=_base('capacity','P17',dict(id='finite_worker_candidates',version=1,scope='isolated_inference_capacity',
        configuration=_configuration(protocol),recipe=r,strategy='explicit_ordered',adaptive_search='not_implemented'),
        dict(model=dict(factory=r['factory'],kwargs=r['model_kwargs'],seed=r['seed'],artifact_sha256=r['artifact_sha256']),
            input=dict(shape=r['input_shape'],recipe='ones_exact_NCHW'),
            environment=dict(device=r['device'],python=platform.python_version(),platform=platform.system(),
                             parent_pid=os.getpid(),workers='raw_evidence.candidates.provenance')))
    data.update(requested=dict(candidates=len(rows)),actual=dict(candidates=len(rows),
        **{c:sum(row['classification']==c for row in rows) for c in ('feasible','infeasible','unknown')}),
        measurements=_summary(rows,protocol),raw_evidence=dict(candidates=rows))
    data['diagnostics']['limitations']=['Trusted importable CPU model factory; no external side effects or subprocesses.',
        'Windows hard private-commit budget, not RSS or physical RAM measurement; other platforms unsupported.',
        'Allocator fraction is not physical GPU capacity. No training-capacity or global maximum claim.',
        'CPU RSS snapshots/lifetime HWM do not measure the exact inference-window peak.',
        'B4 original native fail-fast root cause remains UNRESOLVED; risk OPEN.']
    return CapacityResult(data)


def _validate_capacity(data):
    protocol=data['protocol'];plan=BatchSearchProtocol(**protocol['configuration'])
    if protocol['id']!='finite_worker_candidates' or protocol['scope']!='isolated_inference_capacity':
        raise ValueError('invalid capacity protocol')
    recipe=ColdStartRecipe(**protocol['recipe']).to_dict()
    rows=data['raw_evidence']['candidates']
    if len(rows)!=len(plan.candidates): raise ValueError('candidate count mismatch')
    ids=set();stopped=False
    for row,batch in zip(rows,plan.candidates):
        uuid.UUID(row['candidate_id'])
        if row['candidate_id'] in ids: raise ValueError('duplicate candidate identity')
        ids.add(row['candidate_id'])
        shape=list(recipe['input_shape']);shape[0]=batch
        if row['batch']!=batch or row['shape']!=shape or row['precision']!=recipe['precision'] or row['device']!=recipe['device']:
            raise ValueError('candidate recipe mismatch')
        status=row['status']
        if status not in ('feasible','infeasible_oom','failed','unsupported','resource_limit','crashed','timed_out','not_run'):
            raise ValueError('unknown candidate outcome')
        if row['classification']!=_classification(status): raise ValueError('invalid classification')
        if (status=='not_run')!=stopped: raise ValueError('invalid admission/stop evidence')
        if not row['process_terminated']: raise ValueError('live worker cannot be published')
        if status in ('feasible','infeasible_oom'):
            if row['exit_code']!=0 or type(row['worker_pid']) is not int:
                raise ValueError('successful worker receipt required for feasible/OOM classification')
        if status!='feasible' and not row['reason']: raise ValueError('missing failure/stop reason')
        if status=='feasible':
            if row['exit_code']!=0 or not row['runtime_ready'] or not row['forward_completed'] or row['actual_batch']!=batch:
                raise ValueError('unverified feasible batch')
            p=PrecisionResult(row['sweep']).to_dict()
            if p['provenance']['input']['shape']!=shape or p['protocol']['precision']!=recipe['precision']:
                raise ValueError('mismatched forward evidence')
            conf=p['protocol']['configuration']
            if any(conf[k]!=getattr(plan,k) for k in ('warmup_iterations','measure_iterations','repeats')):
                raise ValueError('mismatched measurement counts')
            if row['latency_mean_ms']!=p['measurements']['latency_mean_ms']['value']:
                raise ValueError('invalid latency evidence')
            if row['memory'] is None or row['provenance'] is None: raise ValueError('missing memory/runtime evidence')
            memory=row['memory']
            if memory['private_commit_limit_bytes']!=plan.cpu_memory_limit_bytes or memory['input_limit_bytes']!=plan.input_limit_bytes:
                raise ValueError('missing resource budget evidence')
            expected_device='cuda:0' if recipe['device']=='cuda' else recipe['device']
            if p['provenance']['input']['device']!=expected_device or row['provenance']['device']!=expected_device:
                raise ValueError('device fallback forbidden')
        elif row['forward_completed'] or row['actual_batch'] is not None or row['sweep'] is not None:
            raise ValueError('failed candidate cannot claim completion')
        if status not in ('feasible','infeasible_oom'): stopped=True
        if status=='infeasible_oom' and plan.stop_policy=='first_infeasible': stopped=True
    if data['measurements']!=_summary(rows,plan): raise ValueError('unsupported maximum/completeness claim')
    expected=dict(candidates=len(rows),**{c:sum(r['classification']==c for r in rows) for c in ('feasible','infeasible','unknown')})
    if data['actual']!=expected or data['requested']!=dict(candidates=len(rows)): raise ValueError('invalid counts')
