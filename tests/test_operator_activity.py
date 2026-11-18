"""Synthetic protocol tests; native kernel evidence lives in the acceptance capture."""
import json
import sqlite3

import pytest

from hyso.performance import analyze_nsys_operators


@pytest.fixture
def trace(tmp_path):
    token='11111111-2222-4333-8444-555555555555';pid=1234
    receipt=tmp_path/'capture.json';path=tmp_path/'trace.sqlite'
    receipt.write_text(json.dumps(dict(format='hyso.performance.p14.nsys_capture',version=1,
        status='completed',terminal_completed=True,measured_forwards=1,output_equal=[True],
        trace_token=token,worker_pid=pid)))
    gp=(1<<48)|(pid<<24);gt=gp|8
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE META_DATA_EXPORT (name TEXT,value TEXT);
            CREATE TABLE StringIds (id INT,value TEXT);
            CREATE TABLE NVTX_EVENTS (start INT,end INT,text TEXT,textId INT,globalTid INT,eventType INT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INT,end INT,globalTid INT,correlationId INT,nameId INT,returnValue INT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT,end INT,deviceId INT,contextId INT,streamId INT,correlationId INT,globalPid INT,shortName INT);
            CREATE TABLE DIAGNOSTIC_EVENT (text TEXT,globalPid INT);
            CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (processId INT,deviceId INT,contextId INT);
        ''')
        db.executemany('INSERT INTO META_DATA_EXPORT VALUES (?,?)',[
            ('EXPORT_PRODUCT_NAME','NVIDIA Nsight Systems'),('EXPORT_PRODUCT_VERSION','test'),('EXPORT_SCHEMA_VERSION','3.31.0')])
        db.executemany('INSERT INTO StringIds VALUES (?,?)',[(1,'cudaLaunchKernel_v7000'),(2,'gemm')])
        db.executemany('INSERT INTO NVTX_EVENTS VALUES (?,?,?,NULL,?,59)',[
            (0,200,f'hyso.p14/{token}/forward/0',gt),
            (10,100,'aten::linear, op_id = 1',gt),(20,80,'aten::addmm, op_id = 2',gt)])
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (30,35,?,42,1,0)',(gt,))
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (150,180,0,1,7,42,?,2)',(gp,))
        db.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES (?,?)',('PyTorch autograd_nvtx injection initialized successfully.',gp))
        db.execute('INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES (?,0,1)',(pid,))
    return path,receipt


def test_real_ids_and_innermost_stack_not_gpu_time_containment(trace):
    path,receipt=trace;result=analyze_nsys_operators(path,capture_path=receipt)
    assert result['status']=='complete' and result['mapped_kernel_count']==1
    assert result['kernel_count']==1 and result['total_kernel_us']==.03
    k=result['kernels'][0]
    assert k['start_ns']>max(o['end'] for o in result['operators'])
    assert k['operator_chain_nvtx_ids']==[2,3] and k['operator_nvtx_id']==3
    assert k['correlation_id']==42 and k['launch_id']==1
    parent,child=result['operators']
    assert parent['exclusive_kernel_count']==0 and parent['attribution_status']=='no_direct_kernel_observed'
    assert child['exclusive_kernel_count']==1
    assert sum(o['exclusive_kernel_us'] for o in result['operators'])==result['total_kernel_us']


@pytest.mark.parametrize('sql',[
    'UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET correlationId=99',
    'INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME',
    'UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid=globalTid+1',
    'UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid=globalTid+16777216',
    'UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET returnValue=1',
    'UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET end=start-1',
    'UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET correlationId=NULL; UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET correlationId=NULL',
    "UPDATE NVTX_EVENTS SET text='aten::addmm' WHERE rowid=3",
    'UPDATE NVTX_EVENTS SET start=10,end=100 WHERE rowid=3',
    'UPDATE NVTX_EVENTS SET start=32 WHERE rowid=3',
    'UPDATE NVTX_EVENTS SET end=NULL WHERE rowid=3',
    'INSERT INTO NVTX_EVENTS SELECT * FROM NVTX_EVENTS WHERE rowid=3',
    "INSERT INTO NVTX_EVENTS SELECT 25,90,'aten::other, op_id = 3',NULL,globalTid,59 FROM NVTX_EVENTS WHERE rowid=3",
    'DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME',
])
def test_missing_or_ambiguous_chain_is_unsupported_without_time_fallback(trace,sql):
    path,receipt=trace
    with sqlite3.connect(path) as db:db.executescript(sql)
    result=analyze_nsys_operators(path,capture_path=receipt)
    assert result['status']=='partial' and result['attribution_status']=='unsupported'
    assert result['kernel_count']==1 and result['total_kernel_us']==.03
    assert result['kernels'][0]['operator_nvtx_id'] is None
    assert result['kernels'][0]['status']=='unsupported'
    assert all(o['exclusive_kernel_count'] is None for o in result['operators'])


def test_missing_gpu_activity_is_null_and_launch_is_reported(trace):
    path,receipt=trace
    with sqlite3.connect(path) as db:db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL')
    result=analyze_nsys_operators(path,capture_path=receipt)
    assert result['kernel_count'] is None and result['total_kernel_us'] is None
    assert result['cuda_activity_status']=='unsupported' and result['missing_kernel_launches']
    assert all(o['exclusive_kernel_us'] is None for o in result['operators'])


@pytest.mark.parametrize('sql',[
    'DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL',
    "UPDATE META_DATA_EXPORT SET value='2.0' WHERE name='EXPORT_SCHEMA_VERSION'",
    'DELETE FROM DIAGNOSTIC_EVENT',
    'UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET contextId=999',
    'UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=149',
    'UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=201',
    'INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL',
    'INSERT INTO NVTX_EVENTS SELECT * FROM NVTX_EVENTS WHERE rowid=1',
])
def test_invalid_evidence_rejected(trace,sql):
    path,receipt=trace
    with sqlite3.connect(path) as db:db.execute(sql)
    with pytest.raises(ValueError):analyze_nsys_operators(path,capture_path=receipt)


def test_receipt_validation_and_capture_identity(trace):
    path,receipt=trace;data=json.loads(receipt.read_text())
    data['output_equal']=[False];receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError):analyze_nsys_operators(path,capture_path=receipt)
