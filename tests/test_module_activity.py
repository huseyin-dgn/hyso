"""Bounded SQLite fixtures; no CUDA execution or synthetic native acceptance."""
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from hyso.performance import analyze_nsys_modules


def pin(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def trace(tmp_path):
    path=tmp_path/'trace.sqlite';report=tmp_path/'trace.nsys-rep'
    report.write_bytes(b'synthetic report identity only')
    gp=(1<<48)|(1234<<24);gt=gp|8
    with sqlite3.connect(path) as db:
        db.executescript('''
        CREATE TABLE META_DATA_EXPORT(name TEXT,value TEXT);
        INSERT INTO META_DATA_EXPORT VALUES('EXPORT_PRODUCT_NAME','NVIDIA Nsight Systems'),('EXPORT_SCHEMA_VERSION','3.31.0');
        CREATE TABLE StringIds(id INT,value TEXT);
        INSERT INTO StringIds VALUES(1,'cudaLaunchKernel_v7000'),(2,'gemm');
        CREATE TABLE NVTX_EVENTS(start INT,end INT,text TEXT,textId INT,globalTid INT,eventType INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INT,end INT,globalTid INT,correlationId INT,nameId INT,returnValue INT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INT,end INT,deviceId INT,contextId INT,streamId INT,correlationId INT,globalPid INT,shortName INT);
        CREATE TABLE TARGET_INFO_CUDA_CONTEXT_INFO(processId INT,deviceId INT,contextId INT);
        INSERT INTO TARGET_INFO_CUDA_CONTEXT_INFO VALUES(1234,0,1);
        CREATE TABLE DIAGNOSTIC_EVENT(severity INT,text TEXT,globalPid INT);
        ''')
        db.executemany('INSERT INTO NVTX_EVENTS VALUES(?,?,?,NULL,?,59)',[
            (0,200,'hyso.p13/forward/0',gt),(10,100,'hyso.p13.module/root',gt),
            (20,80,'hyso.p13.module/proj',gt)])
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(30,35,?,42,1,0)',(gt,))
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(150,180,0,1,7,42,?,2)',(gp,))
        db.executemany('INSERT INTO DIAGNOSTIC_EVENT VALUES(1,?,?)',[
            ('CUDA injection initialized successfully.',gp),('NVTX injection initialized successfully.',gp)])
        db.executescript('ALTER TABLE NVTX_EVENTS ADD COLUMN endGlobalTid INT; ALTER TABLE NVTX_EVENTS ADD COLUMN domainId INT DEFAULT 0;')
    return dict(sqlite=pin(path),report=pin(report),global_pid=gp,forward_labels=['hyso.p13/forward/0'])


def mutate(cfg,sql):
    with sqlite3.connect(cfg['sqlite']['path']) as db: db.executescript(sql)
    cfg['sqlite']=pin(Path(cfg['sqlite']['path']))


def receipt(cfg,tmp_path):
    artifact=tmp_path/'source';artifact.write_bytes(b'fixture source/input/checkpoint')
    data=dict(format='hyso.performance.p13.nsys_capture',version=1,status='completed',
        global_pid=cfg['global_pid'],worker_pid=1234,forward_labels=cfg['forward_labels'],
        sqlite_sha256=cfg['sqlite']['sha256'],report_sha256=cfg['report']['sha256'],
        nsys_exit_code=0,output_equal=[True],cudnn_tf32_disabled=True,gpu_uuid='GPU-fixture',
        authorization=dict(approved=True,basis='unit fixture'),supervisor=dict(owned_process_tree=True,
            process_terminated=True,cleanup=dict(active_processes=0)),
        checkpoint=pin(artifact),input=pin(artifact),worker_source=pin(artifact))
    path=tmp_path/'capture.json';path.write_text(json.dumps(data));cfg['capture']=pin(path)
    return data,path


def test_exact_host_stack_and_gpu_duration_without_receipt(trace):
    before=trace['sqlite'].copy();r=analyze_nsys_modules(trace)
    assert r['status']=='partial' and r['attribution_complete'] and not r['provenance_verified']
    assert r['mapped_kernel_count']==r['measured_kernel_count']==1
    k=r['kernels'][0]
    assert k['module_nvtx_id']==3 and k['module_chain_nvtx_ids']==[2,3]
    assert k['duration_us']==.03 and k['start']>r['modules'][1]['end']
    assert pin(Path(before['path']))==before
    assert r['feature_ids']==['P13']


def test_receipt_completes_only_provenance(trace,tmp_path):
    receipt(trace,tmp_path)
    assert analyze_nsys_modules(trace)['status']=='complete'


@pytest.mark.parametrize('sql',[
    'DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME',
    'UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET correlationId=99',
    'INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME',
    'UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET returnValue=1',
    'UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid=globalTid+16777216',
    'UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET correlationId=NULL',
    "UPDATE StringIds SET value='cudaGraphLaunch' WHERE id=1",
    'UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=25',
    'UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=210',
    'DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL',
    'DELETE FROM NVTX_EVENTS WHERE rowid IN(2,3)',
    'INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL SELECT 151,181,deviceId,contextId,streamId,correlationId,globalPid,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL',
])
def test_incomplete_evidence_cannot_pass(trace,tmp_path,sql):
    mutate(trace,sql);receipt(trace,tmp_path)
    r=analyze_nsys_modules(trace)
    assert r['status']=='partial' and not r['attribution_complete']


@pytest.mark.parametrize('sql',[
    'UPDATE NVTX_EVENTS SET end=NULL WHERE rowid=3',
    'UPDATE NVTX_EVENTS SET start=10,end=100 WHERE rowid=3',
    'UPDATE NVTX_EVENTS SET end=110 WHERE rowid=3',
    'UPDATE NVTX_EVENTS SET end=210 WHERE rowid=3',
    'UPDATE NVTX_EVENTS SET domainId=1 WHERE rowid=3',
    'UPDATE NVTX_EVENTS SET endGlobalTid=globalTid+1 WHERE rowid=3',
    'INSERT INTO NVTX_EVENTS SELECT * FROM NVTX_EVENTS WHERE rowid=1',
    'INSERT INTO NVTX_EVENTS SELECT * FROM NVTX_EVENTS WHERE rowid=3',
    'UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET contextId=2',
    'UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=start',
    'INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL',
    'UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET start=190,end=210',
    'UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid=globalTid+1',
    'DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL',
    "UPDATE META_DATA_EXPORT SET value='2.0' WHERE name='EXPORT_SCHEMA_VERSION'",
])
def test_invalid_evidence_rejected(trace,sql):
    mutate(trace,sql)
    with pytest.raises(ValueError):analyze_nsys_modules(trace)


def test_outside_kernels_counted_without_inflating_measured_coverage(trace):
    mutate(trace,'''INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
        SELECT 250,280,deviceId,contextId,streamId,99,globalPid,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL;''')
    r=analyze_nsys_modules(trace)
    assert r['out_of_scope_kernel_count']==1 and r['measured_kernel_count']==r['mapped_kernel_count']==1


def test_missing_launch_kernel_and_empty_forward(trace):
    mutate(trace,'''INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME SELECT 40,45,globalTid,99,1,0 FROM CUPTI_ACTIVITY_KIND_RUNTIME;
        INSERT INTO NVTX_EVENTS SELECT 210,290,'hyso.p13/forward/1',NULL,globalTid,59,NULL,0 FROM NVTX_EVENTS WHERE rowid=1;''')
    trace['forward_labels'].append('hyso.p13/forward/1')
    r=analyze_nsys_modules(trace)
    assert r['missing_kernel_launches'] and not r['attribution_complete']


@pytest.mark.parametrize('field,value',[
    ('output_equal',[False]),('nsys_exit_code',1),('global_pid',1234),('sqlite_sha256','0'*64),
    ('supervisor',dict(owned_process_tree=True,process_terminated=True,cleanup=dict(active_processes=1))),
    ('authorization',dict(approved=False,basis='denied')),
])
def test_invalid_receipt(trace,tmp_path,field,value):
    data,path=receipt(trace,tmp_path);data[field]=value
    path.write_text(json.dumps(data));trace['capture']=pin(path)
    with pytest.raises(ValueError):analyze_nsys_modules(trace)


def test_hash_mismatch_and_budget(trace):
    trace['report']['sha256']='0'*64
    with pytest.raises(ValueError,match='SHA256'):analyze_nsys_modules(trace)


def test_export_identity_rechecked_after_analysis(trace,monkeypatch):
    from hyso.performance import _module_activity as reader
    original=reader._sha;calls=[]
    def changed(path):
        if str(path)==trace['sqlite']['path']:
            calls.append(path)
            if len(calls)>1:return '0'*64
        return original(path)
    monkeypatch.setattr(reader,'_sha',changed)
    with pytest.raises(ValueError,match='SHA256'):analyze_nsys_modules(trace)


def test_row_budget(trace):
    with pytest.raises(ValueError,match='budget'):analyze_nsys_modules(trace,max_rows=1)


def test_live_sqlite_sidecars_rejected(trace):
    Path(trace['sqlite']['path']+'-wal').write_bytes(b'not a closed export')
    with pytest.raises(ValueError,match='standalone'):analyze_nsys_modules(trace)


def test_target_warning_blocks_but_other_process_warning_does_not(trace):
    mutate(trace,'INSERT INTO DIAGNOSTIC_EVENT VALUES(2,\'missing data\',123)')
    assert analyze_nsys_modules(trace)['attribution_complete']
    mutate(trace,f"INSERT INTO DIAGNOSTIC_EVENT VALUES(2,'missing data',{trace['global_pid']})")
    assert not analyze_nsys_modules(trace)['attribution_complete']
