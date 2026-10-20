"""P15/P16 accounting, OS counters, sensor boundaries and immutable evidence."""
from dataclasses import FrozenInstanceError
import csv
import json
import random
import subprocess
import threading
import time

import numpy as np
import pytest
import torch

from hyso.performance import (TransferProtocol,TransferResult,TransferError,measure_transfers,
    UtilizationProtocol,UtilizationResult,measure_utilization)
from hyso.performance import _telemetry
from hyso.performance.inference import _base
from hyso.performance.transfers import _transfer_metrics
from hyso.performance.utilization import _cpu_intervals


def counter(t,k=0,u=0,status='measured'):
    return dict(timestamp_ns=t,query_start_ns=t,query_end_ns=t,kernel_ns=k,user_ns=u,
        status=status,reason=None,system_status='unsupported')


def test_cpu_normalizations_and_system_idle_exclusion():
    a=counter(0); b=counter(10**9,100000000,900000000)
    a.update(system_status='measured',system_idle_ns=0,system_kernel_ns=0,system_user_ns=0)
    b.update(system_status='measured',system_idle_ns=8*10**9,system_kernel_ns=10*10**9,system_user_ns=6*10**9)
    row=_cpu_intervals([a,b],16,10**9)[0]
    assert row['single_core_percent']==100 and row['machine_capacity_percent']==6.25
    assert row['system_busy_percent']==50 and row['kernel_delta_ns']==100000000
    assert _cpu_intervals([a,b],128,10**9)[0]['system_busy_percent'] is None


@pytest.mark.parametrize('end,status',[(counter(0),'invalid'),(counter(100,-1,0),'counter_reset'),
    (counter(100,status='unsupported'),'missing')])
def test_reset_missing_invalid(end,status):
    r=_cpu_intervals([counter(0),end],4,100)[0]
    assert r['status']==status and r['single_core_percent'] is None


def test_gap_not_interpolated():
    r=_cpu_intervals([counter(1),counter(1001,20,30)],4,100)[0]
    assert r['gap'] and r['wall_ns']==1000 and r['single_core_percent']==5


@pytest.mark.parametrize('busy,expected',[('17',17.),('N/A',None),('[Not Supported]',None)])
def test_sensor_domain_units_and_unavailable_fields(busy,expected):
    r=_telemetry.parse_gpu_csv('GPU-test, 0000:01:00.0, RTX, 591.74, '+busy+', 4, 2, 3, 6, 2026/10/20 00:00:00.000')
    assert r['gpu_busy_percent']==expected and r['memory_used_bytes']==2*1024**2
    assert r['memory_controller_busy_percent']==4


@pytest.mark.parametrize('text',['','bad','GPU-x, x, RTX, 1, 101, 0, 0, 0, 0, stamp',
    'GPU-x, x, RTX, 1, nan, 0, 0, 0, 0, stamp', 'GPU-x, x, RTX, 1, 1, 0, -1, 0, 0, stamp'])
def test_sensor_invalid_not_fake_zero(text):
    with pytest.raises(ValueError):_telemetry.parse_gpu_csv(text)


def test_missing_tool(monkeypatch):
    monkeypatch.setattr(_telemetry.shutil,'which',lambda name:None)
    r=_telemetry.gpu_observe('0',.1)
    assert r['status']=='unsupported' and r['values'] is None


@pytest.mark.parametrize('error',[subprocess.TimeoutExpired('nvidia-smi',.1),PermissionError('denied')])
def test_sensor_query_timeout_permission(monkeypatch,error):
    monkeypatch.setattr(_telemetry.shutil,'which',lambda name:'tool')
    def fail(*a,**kw):raise error
    monkeypatch.setattr(_telemetry.subprocess,'run',fail)
    r=_telemetry.gpu_observe('0',.1)
    assert r['status'] in ('timed_out','unavailable') and r['values'] is None


def test_sensor_native_crash_stops_not_unavailable(monkeypatch):
    monkeypatch.setattr(_telemetry.shutil,'which',lambda name:'tool')
    monkeypatch.setattr(_telemetry.subprocess,'run',lambda *a,**kw:subprocess.CompletedProcess([],0xc0000409,'',''))
    with pytest.raises(RuntimeError,match='native sensor process crash'):_telemetry.gpu_observe('0',.1)


def test_multiple_physical_devices_not_aggregated():
    row='GPU-x, x, RTX, 1, 1, 0, 0, 0, 0, stamp'
    with pytest.raises(ValueError):_telemetry.parse_gpu_csv(row+'\n'+row)


@pytest.fixture(scope='module')
def utilization():
    return measure_utilization(protocol=UtilizationProtocol(duration_seconds=.12,interval_seconds=.02))


def test_real_windows_counter_and_parent_state(utilization):
    d=utilization.to_dict()
    assert d['measurements']['cpu']['status']=='measured'
    assert d['raw_evidence']['cpu_intervals'] and d['actual']['cpu_samples']>=3
    assert d['measurements']['gpu']['tensor_core_utilization']['value'] is None
    assert d['measurements']['gpu']['fields']['gpu_busy_percent']['sample_mean'] is None
    assert d['raw_evidence']['query_overhead_ns']>0


def test_process_workload_counter_reference():
    before=_telemetry.cpu_observe(); until=time.perf_counter()+.06
    n=0
    while time.perf_counter()<until:n+=1
    after=_telemetry.cpu_observe(); row=_cpu_intervals([before,after],1,10**9)[0]
    assert n>0 and row['status']=='measured' and row['user_delta_ns']>0
    assert row['single_core_percent']==100*(row['user_delta_ns']+row['kernel_delta_ns'])/row['wall_ns']


def test_sampler_unavailable_and_budget(monkeypatch):
    def unavailable():raise PermissionError('denied')
    monkeypatch.setattr(_telemetry,'cpu_observe',unavailable)
    r=measure_utilization(protocol=UtilizationProtocol(duration_seconds=.1,interval_seconds=.01,max_samples=2,min_valid_intervals=1)).to_dict()
    assert r['actual']['stop_reason']=='sample_budget'
    assert r['measurements']['cpu']['single_core_percent'] is None and r['measurements']['missing_interval_count']==1


def test_missing_gpu_sensor_preserves_cpu(monkeypatch):
    monkeypatch.setattr(_telemetry.shutil,'which',lambda name:None)
    r=measure_utilization(protocol=UtilizationProtocol(duration_seconds=.08,interval_seconds=.01,gpu=True)).to_dict()
    assert r['measurements']['cpu']['status']=='measured' and r['measurements']['status']=='partial'
    assert r['measurements']['gpu']['fields']['gpu_busy_percent']['sample_mean'] is None
    assert all(s['values'] is None for s in r['raw_evidence']['gpu_samples'])


@pytest.mark.parametrize('kwargs',[dict(duration_seconds=61),dict(interval_seconds=0),dict(max_samples=1),
    dict(min_valid_intervals=128),dict(gpu_selector='--bad'),dict(sensor_timeout_seconds=6),dict(gpu=1)])
def test_utilization_protocol_rejects_invalid(kwargs):
    with pytest.raises((ValueError,TypeError)):UtilizationProtocol(**kwargs)


@pytest.mark.parametrize('kwargs',[dict(shape=()),dict(shape=(0,)),dict(measure_iterations=0),dict(pinned=1),
    dict(directions=('H2D','H2D')),dict(directions=('peer',)),dict(timeout_seconds=601),dict(cuda_memory_fraction=2)])
def test_transfer_protocol_rejects_invalid(kwargs):
    with pytest.raises((ValueError,TypeError)):TransferProtocol(**kwargs)


def transfer_fixture():
    p=TransferProtocol(shape=(4,),directions=('H2D',),warmup_iterations=0,measure_iterations=1,repeats=1)
    from dataclasses import asdict
    tensor=dict(shape=[4],numel=4,element_size=4,dtype='float32',layout='contiguous_dense',transfer_bytes=16)
    rows=[dict(direction='H2D',repeat=0,iteration=0,transfer_bytes=16,host_start_ns=100,
        host_end_ns=200,terminal_end_ns=2000100,host_enqueue_ms=.0001,device_stream_ms=2.,
        terminal_complete=True,equal=True,host_pinned=False,non_blocking_requested=False)]
    d=_base('transfer','P15',dict(id='owned_serial_copy',version=1,scope='fixture',configuration=json.loads(json.dumps(asdict(p)))),
        dict(model={'test':'synthetic'},input=tensor,environment={'device':'cuda:0'}))
    d.update(requested=dict(measured_copies=1),actual=dict(completed_copies=1,warmup_copies=0,validation_copies=1,pending_copies=0),
        measurements=_transfer_metrics(rows),raw_evidence=dict(copies=rows,tensor=tensor))
    return TransferResult(d)


def test_bytes_and_completed_bandwidth():
    d=transfer_fixture().to_dict(); m=d['measurements']
    assert m['directions'][0]['completed_bytes_per_second']==8000
    assert m['overlap']['value'] is None and m['asynchronous_execution']['value'] is None


@pytest.mark.parametrize('field,value',[('transfer_bytes',17),('terminal_complete',False),('equal',False),
    ('host_enqueue_ms',1),('device_stream_ms',float('nan')),('host_pinned',True)])
def test_copy_tampering_rejected(field,value):
    d=transfer_fixture().to_dict();d['raw_evidence']['copies'][0][field]=value
    with pytest.raises(ValueError):TransferResult(d)


def test_zero_event_interval_not_infinite_bandwidth():
    row=transfer_fixture().to_dict()['raw_evidence']['copies'][0];row['device_stream_ms']=0
    assert _transfer_metrics([row])['directions'][0]['completed_bytes_per_second'] is None


def test_exports_immutable_no_clobber_and_csv(utilization,tmp_path):
    for result in (utilization,transfer_fixture()):
        d=result.to_dict();path=tmp_path/(d['result_id']+'.json');csv_path=path.with_suffix('.csv')
        with pytest.raises((FrozenInstanceError,AttributeError)):result._payload='x'
        result.to_json(path);result.to_csv(csv_path)
        assert type(result)(json.loads(path.read_text(encoding='utf-8'))).to_dict()==d
        with csv_path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
        assert rows and json.loads(rows[0]['payload'])
        with pytest.raises(FileExistsError):result.to_json(path)
        with pytest.raises(FileExistsError):result.to_csv(csv_path)
        d['actual'][next(iter(d['actual']))]=-1
        with pytest.raises(ValueError):type(result)(d)


def test_sampler_rng_preservation():
    py=random.getstate();ns=np.random.get_state();rng=torch.get_rng_state();cuda=torch.cuda.is_initialized()
    measure_utilization(protocol=UtilizationProtocol(duration_seconds=.06,interval_seconds=.01))
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],ns[1])
    assert torch.equal(torch.get_rng_state(),rng) and torch.cuda.is_initialized()==cuda


def test_worker_timeout_cleanup_and_no_parent_cuda():
    rng=torch.get_rng_state();cuda=torch.cuda.is_initialized()
    with pytest.raises(TransferError) as error:
        measure_transfers(protocol=TransferProtocol(timeout_seconds=.01))
    assert error.value.diagnostics['status']=='timed_out' and error.value.diagnostics['process_terminated']
    assert torch.equal(rng,torch.get_rng_state()) and torch.cuda.is_initialized()==cuda


def test_worker_native_crash_never_oom(monkeypatch):
    import hyso.performance.transfers as module
    class Process:
        pid=1234;returncode=0xc0000409
        def wait(self,**kwargs):return self.returncode
        def poll(self):return self.returncode
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**kw:Process())
    with pytest.raises(TransferError) as error:measure_transfers()
    assert error.value.diagnostics['status']=='crashed' and error.value.diagnostics['exit_code']==0xc0000409


@pytest.mark.parametrize('change',['counter','clock','metric','count'])
def test_utilization_tampering_rejected(utilization,change):
    d=utilization.to_dict()
    if change=='counter':d['raw_evidence']['cpu_samples'][-1]['user_ns']+=100000000
    if change=='clock':d['raw_evidence']['cpu_samples'][0]['timestamp_ns']=0
    if change=='metric':d['measurements']['cpu']['single_core_percent']=999
    if change=='count':d['actual']['cpu_samples']=0
    with pytest.raises(ValueError):UtilizationResult(d)


@pytest.mark.parametrize('kwargs,status',[({'layout':'sparse'},'unsupported'),({'dtype':'qint8'},'unsupported'),
    ({'device':'cuda:1'},'unsupported'),({'input_limit_bytes':1},'failed')])
def test_worker_unsupported_budget_before_cuda(kwargs,status):
    with pytest.raises(TransferError) as error:measure_transfers(protocol=TransferProtocol(**kwargs))
    assert error.value.diagnostics['status']==status and error.value.diagnostics['process_terminated']


def test_experiment_manager_both_results(utilization,tmp_path):
    from hyso.experiments import ExperimentManager
    store=tmp_path/'runs';m=ExperimentManager(store);e=m.create_experiment('telemetry');m.close()
    ExperimentManager.migrate_store(store);m=ExperimentManager(store);run=m.create_run(e['id'],'benchmark',config={'feature':'P15/P16'})
    run.start();artifacts=[]
    for i,r in enumerate((transfer_fixture(),utilization)):
        path=tmp_path/(r.to_dict()['result_id']+'.json');r.to_json(path)
        artifacts.append(run.add_artifact(path,kind='performance_result',ownership='copy'))
        run.log_metric(name='collection.observation',value=i,unit='count',step=i,step_kind='observation',source='performance.P15_P16')
    rid=run.id;run.finish();m.close();m=ExperimentManager(store)
    assert len(m.get_run(rid).read_metrics())==2
    assert all(m.verify_artifact(a['id'])['status']=='verified' for a in artifacts);m.close()
