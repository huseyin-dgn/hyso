"""Synthetic schema tests validate accounting; only native acceptance proves GPU activity."""
import sqlite3

import pytest

from hyso.performance import TransferProtocol,TransferResult
from hyso.performance._transfer_activity import _activity_metrics
from test_transfer_performance import comparison_fixture


@pytest.fixture
def activity(tmp_path):
    data=comparison_fixture();data['protocol']['configuration']['profile_capture']=True
    token='11111111-2222-4333-8444-555555555555'
    data['raw_evidence']['overlap_probe']['comparison']['trace_token']=token
    pid=4321;data['raw_evidence']['worker']['worker_pid']=pid
    path=tmp_path/'trace.sqlite'
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INT,end INT,deviceId INT,contextId INT,streamId INT,correlationId INT,globalPid INT,bytes INT,copyKind INT,srcKind INT,dstKind INT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT,end INT,deviceId INT,contextId INT,streamId INT,correlationId INT,globalPid INT,shortName INT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INT,end INT,globalTid INT,correlationId INT,returnValue INT);
            CREATE TABLE NVTX_EVENTS (start INT,end INT,text TEXT,textId INT,globalTid INT);
            CREATE TABLE StringIds (id INT,value TEXT);
            CREATE TABLE META_DATA_EXPORT (name TEXT,value TEXT);
            CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INT,name TEXT);
            CREATE TABLE ENUM_CUDA_MEM_KIND (id INT,name TEXT);
            CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO (processId INT,deviceId INT,contextId INT);
        ''')
        db.executemany('INSERT INTO META_DATA_EXPORT VALUES (?,?)',[
            ('EXPORT_PRODUCT_NAME','NVIDIA Nsight Systems'),('EXPORT_PRODUCT_VERSION','test'),('EXPORT_SCHEMA_VERSION','3.31.0')])
        db.execute("INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (1,'CUDA_MEMCPY_KIND_HTOD')")
        db.executemany('INSERT INTO ENUM_CUDA_MEM_KIND VALUES (?,?)',[(1,'CUDA_MEMOPR_MEMORY_KIND_PINNED'),(2,'CUDA_MEMOPR_MEMORY_KIND_DEVICE')])
        db.execute("INSERT INTO StringIds VALUES (1,'ampere_sgemm')")
        db.execute('INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES (?,0,1)',(pid,))
        gp=(1<<48)|(pid<<24);gt=gp|9
        for index,mode in enumerate(('serial','concurrent')):
            base=100+300*index;stream=7+index
            db.execute('INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)',
                (base-10,base+200,f'hyso.p15/{token}/measure/0/{mode}',None,gt))
            correlation=100*index+1
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,0)',(base-5,base-4,gt,correlation))
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,0,1,7,?,?,16,1,1,2)',(base,base+100,correlation,gp))
            for j in range(16):
                correlation+=1;start=base+(100 if mode=='serial' else 40)+2*j
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,0)',(base-3,base-2,gt,correlation))
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,0,1,?,?,?,1)',(start,start+2,stream,correlation,gp))
    return TransferResult(data),path


def test_optional_reader_links_exact_result_and_keeps_original_unsupported(activity):
    result,path=activity;before=result.to_dict()
    timeline=result.nsys_activity_timeline(path)
    assert timeline['source_result_id']==before['result_id'] and timeline['status']=='measured'
    serial,concurrent=timeline['phases']
    assert serial['overlap_ns']==0 and serial['busy_union_ns']==132
    assert concurrent['overlap_ns']==32 and concurrent['busy_union_ns']==100
    assert len(concurrent['activities'])==17
    assert result.to_dict()==before and before['measurements']['overlap']['status']=='unsupported'
    assert not TransferProtocol().profile_capture
    # GPU timestamps differ completely from the fixture's host timing clock.
    assert serial['activities'][0]['start_ns']==100


def test_no_labels_no_unsupported_to_measured_upgrade(tmp_path):
    with pytest.raises(ValueError,match='no external capture labels'):
        TransferResult(comparison_fixture()).nsys_activity_timeline(tmp_path/'absent.sqlite')
    with pytest.raises(ValueError): TransferProtocol(profile_capture=True)
    with pytest.raises(TypeError): TransferProtocol(profile_capture=1)


@pytest.mark.parametrize('sql',[
    'DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL',
    'DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE rowid=1',
    'DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE rowid=1',
    'DELETE FROM NVTX_EVENTS WHERE rowid=1',
    "UPDATE META_DATA_EXPORT SET value='2.0' WHERE name='EXPORT_SCHEMA_VERSION'",
    "UPDATE NVTX_EVENTS SET text='different capture' WHERE rowid=1",
    'UPDATE NVTX_EVENTS SET globalTid=globalTid+16777216',
    'UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET bytes=32',
    'UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET srcKind=0',
    'UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET copyKind=2',
    'UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET streamId=7 WHERE streamId=8',
    'UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET contextId=2',
    'UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=99999 WHERE rowid=1',
    'UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET returnValue=1 WHERE rowid=1',
    'INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE rowid=1',
    'INSERT INTO NVTX_EVENTS SELECT * FROM NVTX_EVENTS WHERE rowid=1',
])
def test_missing_mismatched_or_ambiguous_gpu_evidence_rejected(activity,sql):
    result,path=activity
    with sqlite3.connect(path) as db: db.execute(sql)
    with pytest.raises(ValueError): result.nsys_activity_timeline(path)


def test_duplicate_and_overlapping_activities_are_unioned_before_intersection():
    m=_activity_metrics([(0,10),(0,10),(5,15)],[(3,8),(6,12),(20,22)])
    assert m['copy_active_ns']==15 and m['compute_active_ns']==11
    assert m['overlap_ns']==9 and m['busy_union_ns']==17
    assert m['overlap_intervals_ns']==[[3,12]]
    assert m['copy_active_ns']+m['compute_active_ns']-m['overlap_ns']==m['busy_union_ns']
    assert _activity_metrics([(0,5)],[(5,10)])['overlap_ns']==0
