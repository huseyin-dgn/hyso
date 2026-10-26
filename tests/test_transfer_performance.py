"""P15 protocol/evidence tests. Synthetic events are not native CUDA evidence."""
import copy
import csv
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyso.performance import TransferProtocol,TransferResult,TransferError,measure_transfers
from hyso.performance.transfers import _transfer_metrics, _comparison_metrics
from test_performance_transfer_utilization import transfer_fixture


def current_fixture(*,pending_start=False,pending_end=True,probe=False):
    d=transfer_fixture().to_dict();raw=d['raw_evidence']
    d['protocol']['evidence_version']=2
    d['protocol']['configuration']['probe_overlap']=probe
    raw['copies'][0].update(start_stream_id=7,end_stream_id=7,start_event_index=0,end_event_index=1,
        start_complete_after_enqueue=not pending_start,end_complete_after_enqueue=not pending_end,host_observation_ns=300,
        events_preinitialized=True)
    observation=dict(cpu=dict(status='unsupported',observation=None,reason='synthetic fixture'),
        cuda_allocated_bytes=16,cuda_reserved_bytes=32,cuda_free_bytes=0,cuda_total_bytes=1024)
    raw.update(stream_id=7,worker=dict(status='completed',exit_code=0,process_terminated=True),
        allocator_fraction=None,cpu_memory_limit_bytes=None,
        cleanup=dict(terminal_completed=True,process_owned_resources=True),
        resources=dict(policy='advisory_only; occupancy_is_not_an_execution_gate',before=observation,after=copy.deepcopy(observation)),
        overlap_probe=(dict(direction='H2D',transfer_bytes=16,copy_stream_id=7,compute_stream_id=8,epoch_stream_id=7,
            compute_waited_on_epoch=True,copy_start_ms=1.,copy_end_ms=4.,compute_start_ms=2.,compute_end_ms=6.,
            compute_shape=[256,256],compute_iterations=16,terminal_completed=True,round_trip_equal=True,
            compute_equal=True,host_pinned=False,non_blocking_requested=False,events_preinitialized=True) if probe else None))
    d['measurements']=_transfer_metrics(raw['copies'],evidence_version=2,overlap_probe=raw['overlap_probe'])
    return d


def test_default_quotas_are_absent_and_explicit_limits_still_work():
    p=TransferProtocol()
    assert all(asdict(p)[k] is None for k in ('cpu_memory_limit_bytes','input_limit_bytes','cuda_memory_fraction'))
    assert not p.probe_overlap
    # Neither shape nor an explicit caller limit has an implicit 8/64 MiB ceiling.
    p=TransferProtocol(shape=(32*1024**2,),input_limit_bytes=128*1024**2,
        cpu_memory_limit_bytes=256*1024**2,cuda_memory_fraction=.2)
    assert p.input_limit_bytes==128*1024**2 and p.cuda_memory_fraction==.2


@pytest.mark.parametrize('kwargs',[dict(cpu_memory_limit_bytes=0),dict(input_limit_bytes=0),
    dict(cuda_memory_fraction=0),dict(cuda_memory_fraction=float('nan')),dict(probe_overlap=1)])
def test_invalid_optional_policy(kwargs):
    with pytest.raises((ValueError,TypeError)): TransferProtocol(**kwargs)


def test_bandwidth_units_and_distinct_host_clocks():
    d=current_fixture();m=TransferResult(d).to_dict()['measurements']
    summary=m['directions'][0]
    assert summary['bytes']==16 and summary['completed_copies']==1 and summary['device_stream_ms']==2.
    assert summary['completed_bytes_per_second']==8000.
    assert summary['effective_bandwidth_gb_s']==8000./1e9
    assert summary['host_enqueue_ms']==.0001 and summary['host_completion_wall_ms']==2.
    assert summary['timing_unit']=='ms' and summary['bandwidth_unit']=='GB/s (10^9 bytes/s)'
    assert m['transfer_status']=='measured'
    # A fully occupied device is advisory, not a reason to reject a completed copy.
    assert d['raw_evidence']['resources']['before']['cuda_free_bytes']==0


@pytest.mark.parametrize('nonblocking',[False,True])
def test_pending_terminal_alone_and_nonblocking_flag_do_not_prove_asynchrony(nonblocking):
    d=current_fixture();d['protocol']['configuration']['non_blocking']=nonblocking
    d['raw_evidence']['copies'][0]['non_blocking_requested']=nonblocking
    m=d['measurements']
    assert m['asynchronous_execution']['status']=='unsupported' and m['asynchronous_execution']['value'] is None
    TransferResult(d)


def test_pending_start_is_narrow_host_asynchrony_evidence_only():
    d=current_fixture(pending_start=True);m=TransferResult(d).to_dict()['measurements']
    assert m['asynchronous_execution']['value'] is True
    assert m['asynchronous_execution']['status']=='observed'
    assert m['asynchronous_execution']['observed_copy_indices']==[0]
    assert m['overlap']['status']=='unsupported' and m['overlap']['value'] is None


def test_event_window_intersection_does_not_prove_hardware_overlap_or_change_bandwidth():
    d=current_fixture(probe=True);m=TransferResult(d).to_dict()['measurements']
    assert m['overlap']['stream_window_intersection_ms']==2.
    assert m['overlap']['probe_status']=='observed'
    assert m['overlap']['value'] is None and m['overlap']['status']=='unsupported'
    assert m['directions']==current_fixture()['measurements']['directions']


def test_zero_duration_has_no_synthetic_bandwidth():
    d=current_fixture();raw=d['raw_evidence'];raw['copies'][0]['device_stream_ms']=0.
    d['measurements']=_transfer_metrics(raw['copies'],evidence_version=2)
    m=TransferResult(d).to_dict()['measurements']
    assert m['directions'][0]['completed_bytes_per_second'] is None
    assert m['directions'][0]['effective_bandwidth_gb_s'] is None and m['transfer_status']=='partial'


@pytest.mark.parametrize('mutate',[
    lambda d:d['protocol'].update(evidence_version=True),
    lambda d:d['raw_evidence']['copies'][0].update(start_stream_id=8),
    lambda d:d['raw_evidence']['copies'][0].update(end_event_index=0),
    lambda d:d['raw_evidence']['copies'][0].update(host_observation_ns=0),
    lambda d:d['raw_evidence']['copies'][0].update(start_complete_after_enqueue=0),
    lambda d:d['raw_evidence']['copies'][0].update(terminal_complete=False),
    lambda d:d['raw_evidence']['copies'][0].update(events_preinitialized=False),
    lambda d:d['raw_evidence']['worker'].update(process_terminated=False),
    lambda d:d['raw_evidence']['cleanup'].update(terminal_completed=False),
    lambda d:d['measurements']['directions'][0].update(effective_bandwidth_gb_s=1.),
    lambda d:d['measurements']['asynchronous_execution'].update(value=True),
    lambda d:d['raw_evidence']['resources']['before'].update(cuda_free_bytes=2048),
])
def test_current_evidence_tamper_rejected(mutate):
    d=current_fixture();mutate(d)
    with pytest.raises(ValueError): TransferResult(d)


@pytest.mark.parametrize('change',[
    dict(compute_stream_id=7),dict(compute_waited_on_epoch=False),dict(round_trip_equal=False),
    dict(compute_equal=False),dict(terminal_completed=False),dict(copy_end_ms=0.),dict(compute_start_ms=float('nan')),
    dict(events_preinitialized=False),
])
def test_probe_stream_order_completion_and_validation_are_required(change):
    d=current_fixture(probe=True);d['raw_evidence']['overlap_probe'].update(change)
    with pytest.raises(ValueError): TransferResult(d)


@pytest.mark.parametrize('legacy',[True,False])
def test_old_and_current_json_csv_reopen_without_migration(legacy,tmp_path):
    d=transfer_fixture().to_dict() if legacy else current_fixture(probe=True)
    if legacy:
        cfg=d['protocol']['configuration'];cfg.pop('probe_overlap',None)
        cfg.update(cpu_memory_limit_bytes=3*1024**3,input_limit_bytes=8*1024**2,cuda_memory_fraction=.1)
    result=TransferResult(d)
    result.to_json(tmp_path/'transfer.json');result.to_csv(tmp_path/'transfer.csv')
    assert TransferResult(json.loads((tmp_path/'transfer.json').read_text())).to_dict()==d
    with (tmp_path/'transfer.csv').open(newline='',encoding='utf-8') as stream:
        assert [json.loads(r['payload']) for r in csv.DictReader(stream)]==d['raw_evidence']['copies']


@pytest.mark.parametrize('status,reason',[
    ('cuda_oom','OutOfMemoryError: controlled CUDA OOM'),
    ('failed','RuntimeError: controlled CUDA error'),
])
def test_worker_error_receipt_is_not_hidden_or_replaced(monkeypatch,status,reason):
    import hyso.performance.transfers as module
    class Process:
        pid=1234;returncode=0
        def wait(self,**kwargs): return 0
        def poll(self): return 0
    def launch(command,**kwargs):
        # Exercise the real parent transport/cleanup, with a controlled error body.
        Path(command[5]).write_text(json.dumps(dict(worker_pid=1234,status=status,reason=reason,result=None)))
        Path(command[6]).write_text(json.dumps(dict(pid=1234,phase='runtime_prepared')))
        return Process()
    monkeypatch.setattr(module.subprocess,'Popen',launch)
    with pytest.raises(TransferError) as error: measure_transfers()
    assert error.value.diagnostics['status']==status and error.value.diagnostics['reason']==reason
    assert error.value.diagnostics['process_terminated']


def test_advisory_resource_observation_does_not_swallow_cuda_errors(monkeypatch):
    from hyso.performance import memory
    from hyso.performance._transfer_worker import _resources
    monkeypatch.setattr(memory,'_observe',lambda:dict(rss_bytes=1,lifetime_hwm_bytes=1))
    def fail(device): raise RuntimeError('controlled CUDA query error')
    with pytest.raises(RuntimeError,match='controlled CUDA query error'):
        _resources(SimpleNamespace(cuda=SimpleNamespace(mem_get_info=fail)))


def comparison_fixture():
    d=current_fixture(probe=True)
    d['protocol']['evidence_version']=3
    d['protocol']['configuration'].update(pinned=True,non_blocking=True,compare_execution=True)
    raw=d['raw_evidence'];raw['copies'][0].update(host_pinned=True,non_blocking_requested=True)
    probe=raw['overlap_probe'];probe.update(host_pinned=True,non_blocking_requested=True)
    def row(stream,end):
        return dict(host_start_ns=100,terminal_end_ns=end,completion_wall_ms=(end-100)/1e6,
            terminal_completed=True,round_trip_equal=True,compute_equal=True,events_preinitialized=True,
            host_pinned=True,non_blocking_requested=True,copy_stream_id=7,compute_stream_id=stream,
            transfer_bytes=16,compute_shape=[256,256],compute_iterations=16)
    probe['comparison']=dict(warmup_pairs=0,pairs=[dict(index=0,order=['serial','concurrent'],
        serial=row(7,2000100),concurrent=row(8,1000100))])
    d['measurements']=_transfer_metrics(raw['copies'],evidence_version=3,overlap_probe=probe)
    return d


@pytest.mark.parametrize('kwargs',[dict(compare_execution=1),dict(compare_execution=True),
    dict(compare_execution=True,probe_overlap=True,pinned=True)])
def test_comparison_requires_explicit_pinned_nonblocking_probe(kwargs):
    with pytest.raises((ValueError,TypeError)): TransferProtocol(**kwargs)


def test_comparison_is_optional_and_preserves_copy_accounting_and_exports(tmp_path):
    assert not TransferProtocol().compare_execution
    d=comparison_fixture();result=TransferResult(d)
    m=d['measurements']['execution_comparison']
    assert m['observed_speedup']==2 and m['observed_time_reduction_percent']==50
    assert not m['reliable_improvement'] and not m['hardware_overlap_proven']
    assert m['pinning_benefit'].startswith('not_isolated')
    assert d['measurements']['directions'][0]['bytes']==16
    assert d['measurements']['overlap']['value'] is None
    result.to_json(tmp_path/'comparison.json');result.to_csv(tmp_path/'comparison.csv')
    assert TransferResult(json.loads((tmp_path/'comparison.json').read_text())).to_dict()==d
    with (tmp_path/'comparison.csv').open(newline='') as f:
        assert [json.loads(r['payload']) for r in csv.DictReader(f)]==d['raw_evidence']['copies']


@pytest.mark.parametrize('change',[
    dict(compute_equal=False),dict(terminal_completed=False),dict(round_trip_equal=False),
    dict(completion_wall_ms=.5),dict(terminal_end_ns=0),dict(host_pinned=False),
    dict(non_blocking_requested=False),dict(compute_stream_id=7),dict(transfer_bytes=32),
    dict(compute_iterations=15),dict(events_preinitialized=False),
])
def test_comparison_rejects_missing_completion_or_unequal_work(change):
    d=comparison_fixture();d['raw_evidence']['overlap_probe']['comparison']['pairs'][0]['concurrent'].update(change)
    with pytest.raises(ValueError): TransferResult(d)


@pytest.mark.parametrize('scenario',['stable_gain','no_gain','regression','noisy','too_few','no_warmup','order_bias'])
def test_recommendation_requires_repeatable_gain_in_both_orders(scenario):
    pairs=[]
    for i in range(8 if scenario!='too_few' else 2):
        serial=10.
        concurrent={'no_gain':10.,'regression':12.,'noisy':(2. if i%2 else 12.),
            'order_bias':(5. if i%2 else 11.)}.get(scenario,7.)
        pairs.append(dict(order=['serial','concurrent'] if i%2==0 else ['concurrent','serial'],
            serial=dict(completion_wall_ms=serial),concurrent=dict(completion_wall_ms=concurrent)))
    m=_comparison_metrics(dict(warmup_pairs=0 if scenario=='no_warmup' else 2,pairs=pairs))
    assert m['reliable_improvement'] is (scenario=='stable_gain')
    assert m['recommendation'].startswith('use_concurrent') is (scenario=='stable_gain')


def test_comparison_decision_and_version_cannot_be_forged():
    d=comparison_fixture();d['measurements']['execution_comparison']['reliable_improvement']=True
    with pytest.raises(ValueError): TransferResult(d)
    d=comparison_fixture();d['protocol']['evidence_version']=2
    with pytest.raises(ValueError): TransferResult(d)
