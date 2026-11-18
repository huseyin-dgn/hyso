"""CPU-only owned-worker breadcrumbs; native-like exits are simulations."""
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import uuid
from types import SimpleNamespace

import pytest
import torch

from hyso.performance import (TrainingMemoryRecipe, IsolatedTrainingMemoryProtocol,
    measure_training_memory_isolated)
from hyso.performance import isolated_training_memory as isolated
from hyso.performance._training_memory_worker import _PhaseTrace, _trace_hooks
from hyso.performance._training_process import _bounded_read, _launch
from hyso.performance import _training_memory_worker as bootstrap


TESTS = Path(__file__).parent.resolve()
SOURCE = TESTS.parent


def protocol(**kwargs):
    return IsolatedTrainingMemoryProtocol(process_memory_bytes=4*1024**3,
        job_memory_bytes=8*1024**3, **kwargs)


def launch(tmp_path, *, hooks=False, failure=None, budget=None, factory=None):
    root = tmp_path/'worker'
    root.mkdir()
    trace = tmp_path/'phases.jsonl'
    recipe = TrainingMemoryRecipe(factory or 'training_memory_factories:tiny',
        factory_search_path=tmp_path if factory else TESTS, epochs=2, seed=71,
        kwargs=dict(optimizer='adamw', failure=failure, state_path=str(tmp_path/'state.json')))
    p = protocol(saved_activation_lifetime=hooks)
    request = dict(token=str(uuid.uuid4()), recipe=recipe.to_dict(), protocol=asdict(p),
        package_root=str(SOURCE), torch_version=str(torch.__version__),
        _phase_trace=dict(path=str(trace), **(budget or {})))
    receipt, paths = _launch(root, request, p)
    worker = _bounded_read(paths['result'],p.max_result_bytes) if paths['result'].exists() else None
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    return receipt, worker, rows, request


@pytest.mark.parametrize('hooks',[False,True])
def test_owned_cpu_trace_phases_state_and_cleanup(hooks,tmp_path):
    assert not torch.cuda.is_initialized()
    receipt,worker,rows,request = launch(tmp_path,hooks=hooks)
    assert isolated._classify(receipt,worker,request['token'])[0]=='success',worker
    assert receipt['exit_code']==0 and receipt['job']['active_processes']==0
    assert any('no measurement authority' in s for s in worker['measurement']['source_p19']['limitations'])
    assert all(set(r)=={'phase','monotonic_ns','pid','run_id','cpu_status'} for r in rows)
    assert {r['pid'] for r in rows}=={receipt['launcher_pid']}
    assert {r['run_id'] for r in rows}=={request['token']}
    assert [r['monotonic_ns'] for r in rows]==sorted(r['monotonic_ns'] for r in rows)
    pairs=[(r['phase'],r['cpu_status']) for r in rows]
    assert pairs[:4]==[('worker_started','end'),('job_admitted','end'),
                        ('torch_import','begin'),('torch_import','end')]
    assert ('cuda_setup','skipped') in pairs
    for phase in ('factory','forward','loss_metrics','backward','scaler_step_attempt',
                  'validation','recorder_cleanup','workload','result_publication'):
        assert pairs.index((phase,'begin')) < pairs.index((phase,'end'))
    for phase in ('saved_hook_context','saved_hook_exit','activation_cleanup'):
        assert (phase,'end' if hooks else 'skipped') in pairs
    assert pairs.index(('saved_hook_exit','end' if hooks else 'skipped')) < pairs.index(('activation_cleanup','end' if hooks else 'skipped'))
    assert pairs[-2:]==[('result_publication','begin'),('result_publication','end')]
    # An independent untraced fit uses the same existing tiny recipe. Compare
    # actual final state, not timestamps affected by fsync and hook overhead.
    state=tmp_path/'untraced-state.json'
    control=TrainingMemoryRecipe('training_memory_factories:tiny',factory_search_path=TESTS,
        epochs=2,seed=71,kwargs=dict(optimizer='adamw',state_path=str(state)))
    result=measure_training_memory_isolated(control,protocol=protocol()).to_dict()
    assert result['outcome']=='success',result['reason']
    assert json.loads(state.read_text())==json.loads((tmp_path/'state.json').read_text())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('failure,phase',[('constructor','factory'),('forward','forward'),
    ('after_forward','forward'),('backward','backward')])
def test_exception_retains_prefix_and_cleanup_without_false_success(failure,phase,tmp_path):
    receipt,worker,rows,request=launch(tmp_path,hooks=True,failure=failure)
    assert isolated._classify(receipt,worker,request['token'])[0]=='workload_exception'
    assert worker['measurement'] is None and receipt['job']['active_processes']==0
    pairs=[(r['phase'],r['cpu_status']) for r in rows]
    assert (phase,'begin') in pairs and (phase,'end') not in pairs
    assert ('workload','error') in pairs
    if failure!='constructor':
        assert ('saved_hook_exit','end') in pairs
        assert ('activation_cleanup','end') in pairs
        assert ('recorder_cleanup','end') in pairs


@pytest.mark.parametrize('during_training',[True,False])
def test_cpu_abrupt_exit_keeps_durable_prefix_and_cannot_succeed(during_training,tmp_path):
    # No fail-fast/driver/native callback: terminate this owned CPU test child
    # with an ordinary nonzero exit at the existing tiny model's forward entry.
    (tmp_path/'abrupt_factory.py').write_text(
        'import os,sys\nsys.path.append('+repr(str(TESTS))+')\n'
        'from training_memory_factories import tiny\n'
        'def abrupt(**kwargs):\n'
        '    workload=tiny(**kwargs)\n'
        '    def stop(model,*args):\n'
        '        if model.training is '+repr(during_training)+': os._exit(23)\n'
        '    target=workload["trainer"].'+('model' if during_training else 'evaluation_model')+'\n'
        '    target.register_forward_pre_hook(stop)\n'
        '    return workload\n',encoding='utf-8')
    receipt,worker,rows,request=launch(tmp_path,factory='abrupt_factory:abrupt')
    assert receipt['exit_code']==23 and worker is None
    assert isolated._classify(receipt,worker,request['token'])[0]=='native_crash'
    assert receipt['job']['active_processes']==0
    assert (rows[-1]['phase'],rows[-1]['cpu_status'])==('forward' if during_training else 'validation','begin')
    assert not (tmp_path/'state.json').exists()


@pytest.mark.parametrize('budget',[dict(max_records=1),dict(max_bytes=1)])
def test_trace_budget_fault_runs_cleanup_and_denies_success(budget,tmp_path):
    receipt,worker,rows,request=launch(tmp_path,hooks=True,budget=budget)
    assert isolated._classify(receipt,worker,request['token'])[0]=='resource_limit'
    assert worker['measurement'] is None and receipt['job']['active_processes']==0
    assert (tmp_path/'state.json').is_file()  # Tracing did not interrupt fit/unwind.
    assert len(rows)<=budget.get('max_records',512)
    assert (tmp_path/'phases.jsonl').stat().st_size<=budget.get('max_bytes',65536)


def test_public_path_has_no_trace_opt_in_or_output(tmp_path,monkeypatch):
    original=isolated._launch
    seen=[]
    def capture(root,request,p):
        seen.append(dict(request))
        return original(root,request,p)
    monkeypatch.setattr(isolated,'_launch',capture)
    assert all('trace' not in f.name for f in fields(IsolatedTrainingMemoryProtocol))
    result=measure_training_memory_isolated(TrainingMemoryRecipe('training_memory_factories:tiny',
        factory_search_path=TESTS,kwargs=dict(failure='constructor')),protocol=protocol())
    assert result.to_dict()['outcome']=='workload_exception'
    assert seen and '_phase_trace' not in seen[0]
    assert _trace_hooks(None, sentinel:=object(),True) is sentinel


def writer(tmp_path,**kwargs):
    return _PhaseTrace(dict(path=str(tmp_path/'unit.jsonl'),**kwargs),str(uuid.uuid4()),SOURCE,tmp_path/'worker')


def test_trace_io_failure_latches_without_interrupting_hook_cleanup(tmp_path,monkeypatch):
    trace=writer(tmp_path)
    monkeypatch.setattr(os,'fsync',lambda *args: (_ for _ in ()).throw(OSError('test fsync')))
    events=[]
    class Hooks:
        def __enter__(self):events.append('enter')
        def __exit__(self,*args):events.append('exit')
    try:
        with pytest.raises(ValueError,match='original workload'):
            with _trace_hooks(trace,Hooks(),True):raise ValueError('original workload')
        assert events==['enter','exit']
        with pytest.raises(RuntimeError,match='trace'):trace.check()
    finally:trace.close()


@pytest.mark.parametrize('config',[dict(max_records=True),dict(max_bytes=0),dict(max_bytes=1048577),
    dict(max_records=4097),dict(unknown=True),dict(path='relative.jsonl')])
def test_trace_rejects_invalid_config(config,tmp_path):
    with pytest.raises((ValueError,TypeError)):
        _PhaseTrace(dict(dict(path=str(tmp_path/'unit.jsonl')),**config),str(uuid.uuid4()),SOURCE,tmp_path/'worker')


def test_trace_never_overwrites_or_writes_inside_source_or_transient_root(tmp_path):
    path=tmp_path/'unit.jsonl';path.write_bytes(b'previous evidence')
    with pytest.raises(FileExistsError):writer(tmp_path)
    assert path.read_bytes()==b'previous evidence'
    for root in (SOURCE,tmp_path):
        with pytest.raises(ValueError,match='persist outside'):
            _PhaseTrace(dict(path=str(root/'forbidden.jsonl')),str(uuid.uuid4()),SOURCE,tmp_path)


def test_invalid_event_or_owner_is_not_serialized(tmp_path,monkeypatch):
    trace=writer(tmp_path)
    try:
        trace.emit('tensor secret','begin')
        with pytest.raises(RuntimeError):trace.check()
        assert (tmp_path/'unit.jsonl').read_bytes()==b''
    finally:trace.close()
    other=tmp_path/'other';other.mkdir();trace=writer(other)
    try:
        monkeypatch.setattr(os,'getpid',lambda:trace.pid+1)
        trace.emit('forward','begin')
        with pytest.raises(RuntimeError):trace.check()
        assert (other/'unit.jsonl').read_bytes()==b''
    finally:trace.close()


def bootstrap_request(tmp_path):
    root=tmp_path/'bootstrap';root.mkdir()
    token=str(uuid.uuid4())
    request=dict(token=token,package_root=str(SOURCE),runtime_paths=[],
        protocol=dict(max_result_bytes=4096),_phase_trace=dict(path=str(tmp_path/'bootstrap.jsonl')))
    (root/'request.json').write_text(json.dumps(request))
    (root/'admit.json').write_text(json.dumps(dict(token=token,pid=os.getpid())))
    return root,request


def test_no_trace_or_import_before_valid_job_admission(tmp_path,monkeypatch):
    root,request=bootstrap_request(tmp_path)
    (root/'admit.json').write_text(json.dumps(dict(token='wrong',pid=os.getpid())))
    monkeypatch.setattr(bootstrap.sys,'argv',['worker',str(root)])
    def forbidden(*args,**kwargs):raise AssertionError('import before admission')
    monkeypatch.setattr(bootstrap.importlib,'import_module',forbidden)
    with pytest.raises(RuntimeError,match='invalid OS admission'):bootstrap.main()
    assert not (tmp_path/'bootstrap.jsonl').exists()


@pytest.mark.parametrize('when',['publication_begin','publication_end','close'])
def test_late_trace_fault_cannot_return_success_exit(when,tmp_path,monkeypatch):
    root,request=bootstrap_request(tmp_path)
    monkeypatch.setattr(bootstrap.sys,'argv',['worker',str(root)])
    monkeypatch.setattr(bootstrap.sys,'path',list(bootstrap.sys.path))
    monkeypatch.setattr(bootstrap.importlib,'import_module',lambda *args:SimpleNamespace(
        _run_recipe=lambda *args,**kwargs:{}))
    original_emit=bootstrap._PhaseTrace.emit
    original_close=bootstrap._PhaseTrace.close
    def emit(self,phase,status,**kwargs):
        if (phase,status)==('result_publication','begin' if when=='publication_begin' else 'end') and when!='close':
            self.fault=OSError
        original_emit(self,phase,status,**kwargs)
    def close(self):
        original_close(self)
        if when=='close':self.fault=OSError
    monkeypatch.setattr(bootstrap._PhaseTrace,'emit',emit)
    monkeypatch.setattr(bootstrap._PhaseTrace,'close',close)
    with pytest.raises(RuntimeError,match='trace'):bootstrap.main()
    # Even a published success is rejected by the unchanged parent OS-exit gate.
    result=json.loads((root/'result.json').read_text()) if (root/'result.json').exists() else None
    assert isolated._classify(dict(exit_code=1,monitor_outcome=None,launcher_pid=os.getpid()),
        result,request['token'])[0]!='success'
