"""Real provider ledger contracts; offline only, never launch ORT or CUDA."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock
from full_acceptance.handlers import _trained_cuda_ort_execution_confirmed


def fixture():
    data=dict(analysis=dict(status='expected_execution_observed'),evidence=dict(
        protocol=dict(warmup=3,runs=10),report=dict(status='completed',completed_runs=13,
            capture_context=dict(run_intervals=[dict(index=i,phase='warmup' if i<3 else 'measured') for i in range(13)]))))
    diag=dict(completed_runs=13,events=[dict(provider='CUDAExecutionProvider')],failure_kind=None,
        observed_execution_providers=['CUDAExecutionProvider'],evidence_assessment=dict(
            node_coverage=dict(status='complete',source_node_count=30,missing_or_partial_nodes=[])))
    return data,diag,SimpleNamespace(warmup=3,runs=10)


def accepts(data,diag,protocol):
    return _trained_cuda_ort_execution_confirmed(Mock(to_dict=lambda:data),diag,protocol)


def test_warmup_plus_measured_ledger_is_accepted():
    data,diag,p=fixture()
    assert accepts(data,diag,p)
    assert 'node_coverage' not in diag


def test_warmup_cannot_replace_missing_measured_calls():
    data,diag,p=fixture();data['evidence']['report']['completed_runs']=diag['completed_runs']=10
    data['evidence']['report']['capture_context']['run_intervals']=data['evidence']['report']['capture_context']['run_intervals'][:10]
    assert not accepts(data,diag,p)


def test_wrong_phase_or_index_is_rejected():
    for field,value in [('phase','warmup'),('index',999)]:
        data,diag,p=fixture();data['evidence']['report']['capture_context']['run_intervals'][-1][field]=value
        assert not accepts(data,diag,p)


def test_CPU_only_or_missing_events_is_rejected():
    for update in [dict(observed_execution_providers=['CPUExecutionProvider']),dict(events=[])]:
        data,diag,p=fixture();diag.update(update);assert not accepts(data,diag,p)


def test_partial_or_missing_coverage_is_rejected():
    for coverage in [{},dict(status='partial',source_node_count=30),dict(status='complete',source_node_count=0),
                     dict(status='complete',source_node_count=30,missing_or_partial_nodes=['node'])]:
        data,diag,p=fixture();diag['evidence_assessment']['node_coverage']=coverage
        assert not accepts(data,diag,p)


def test_failure_or_protocol_drift_is_rejected():
    data,diag,p=fixture();diag['failure_kind']='runtime';assert not accepts(data,diag,p)
    data,diag,p=fixture();data['evidence']['protocol']['runs']=9;assert not accepts(data,diag,p)
    data,diag,p=fixture();data['analysis']['status']='unavailable';assert not accepts(data,diag,p)
