"""Feature acceptance after implementation; legacy 253 tests are retained."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from hyso.performance import WorkerTaskDataLoader, WorkerTaskAnalysisResult
from hyso.performance.dataloader import _hash
from hyso.performance.worker_task_result import _derive


@pytest.fixture(scope='module')
def task_case(tmp_path_factory):
    if os.name!='nt':pytest.skip('Windows Job acceptance')
    folder=tmp_path_factory.mktemp('p21-tasks')
    root=Path(__file__).resolve().parent
    cache={}
    def execute(case):
        if case not in cache:
            output=folder/(case+'-'+str(time.time_ns())+'.json')
            command=[sys.executable,str(root/'p21_native_supervisor.py'),'--script',
                str(root/'p21_task_acceptance.py'),'--case',case,'--output',str(output),'--timeout','180']
            result=subprocess.run(command,capture_output=True,text=True,timeout=195)
            logs='\n'.join(p.read_text(encoding='utf-8',errors='replace') for p in folder.glob(output.stem+'.*.log'))
            assert result.returncode==0,result.stdout+result.stderr+logs
            receipt=json.loads(output.with_suffix('.receipt.json').read_text())
            assert receipt['exit_code']==0 and not receipt['timed_out']
            cache[case]=json.loads(output.read_text(encoding='utf-8'))
        return copy.deepcopy(cache[case])
    return execute


@pytest.mark.parametrize('case',['persistent','nonpersistent'])
def test_real_task_identity_collate_dispatch_rng_and_source(task_case,case):
    task_case(case)


@pytest.mark.parametrize('case',['crash','hang','collate_exception','saturation'])
def test_real_task_failures_have_bounded_cleanup_and_no_complete(task_case,case):
    task_case(case)


@pytest.mark.parametrize('damage',['execution','epoch','batch','token','pid','duplicate','missing',
    'collate','counter','parent_row','clock','lifecycle','drop'])
def test_corrupted_raw_tasks_cannot_be_complete(task_case,damage):
    data=task_case('persistent')
    packet=data['packets'][0]
    if damage=='execution':packet['receipt']['execution_id']='foreign'
    elif damage=='epoch':packet['receipt']['epoch_id']+=1
    elif damage=='batch':packet['receipt']['batch_id']+=1
    elif damage=='token':packet['receipt']['token']='foreign'
    elif damage=='pid':packet['receipt']['worker_pid']+=1
    elif damage=='duplicate':data['packets'].append(copy.deepcopy(packet))
    elif damage=='missing':data['packets'].pop()
    elif damage=='collate':packet['collate']['duration_ns']+=1
    elif damage=='counter':data['trace_counters']['accepted']-=1
    elif damage=='parent_row':next(r for r in data['ledger'] if r['kind']=='deliver')['observation']['epoch_id']+=1
    elif damage=='clock':packet['clock']['monotonic']=False
    elif damage=='lifecycle':data['ledger'][0]['sequence']=99
    elif damage=='drop':packet['samples'][0]['dropped_events']=1
    assert _derive(data)['status']=='incomplete'


def test_v2_roundtrip_integrity_and_no_clobber(task_case,tmp_path):
    data=task_case('persistent')
    result=WorkerTaskAnalysisResult(data)
    path=tmp_path/'tasks.json';result.to_json(path)
    assert WorkerTaskAnalysisResult(json.loads(path.read_text())).to_dict()==data
    with pytest.raises(FileExistsError):result.to_json(path)
    detached=result.to_dict();detached['packets'].clear();assert result.to_dict()==data
    with pytest.raises((AttributeError,TypeError)):result._payload='bad'
    for field in ('binding','packets','analysis'):
        damaged=copy.deepcopy(data)
        if field=='binding':damaged[field]['source_segment_id']='foreign'
        elif field=='packets':damaged[field][0]['collate']['duration_ns']+=1
        else:damaged[field]['status']='fabricated'
        with pytest.raises(ValueError):WorkerTaskAnalysisResult(damaged)


def test_diagnostics_use_worker_local_service_and_parent_local_residence(task_case):
    data=task_case('persistent')
    result=WorkerTaskAnalysisResult(data)
    diagnostics=result.diagnostics()
    assert diagnostics['status']=='partial'
    service=diagnostics['worker_service']
    assert service['status']=='measured' and service['count']==8
    assert service['mean_ms']>=diagnostics['stages']['collate']['mean_ms']>=2
    assert diagnostics['parent_buffer_residence']['status']=='measured'
    assert diagnostics['parent_buffer_residence']['mean_ms']>=0
    assert diagnostics['cross_process_queue_wait']['value'] is None
    assert diagnostics['queue_occupancy']['value'] is None
    assert diagnostics['prefetch_efficiency']['value'] is None
    assert result.to_dict()==data
    # Loss cannot turn the remaining worker packets into a complete result.
    data['packets'].pop()
    data['analysis']=_derive(data)
    data['evidence_sha256']=_hash({k:v for k,v in data.items() if k not in ('analysis','evidence_sha256')})
    incomplete=WorkerTaskAnalysisResult(data).diagnostics()
    assert incomplete['worker_service']['mean_ms'] is None
    assert incomplete['parent_buffer_residence']['mean_ms'] is None


def test_absent_explicit_stage_receipts_are_unsupported(task_case):
    data=task_case('persistent')
    for packet in data['packets']:
        for sample in packet['samples']:
            sample['events']=[r for r in sample['events'] if r['stage']=='dataset_fetch']
    data['analysis']=_derive(data)
    data['evidence_sha256']=_hash({k:v for k,v in data.items() if k not in ('analysis','evidence_sha256')})
    diagnostics=WorkerTaskAnalysisResult(data).diagnostics()
    assert diagnostics['worker_service']['status']=='measured'
    for stage in ('decode','transform'):
        assert diagnostics['stages'][stage]==dict(status='unsupported',count=0,mean_ms=None)


def test_duplicate_epoch_batch_ids_with_distinct_tokens_fail_closed(task_case):
    data=task_case('persistent')
    submits=[r for r in data['ledger'] if r['kind']=='submit']
    submits[1]['batch_id']=submits[0]['batch_id']
    assert 'duplicate epoch/batch task identity' in _derive(data)['errors']


def test_unsupported_loader_scopes_rejected():
    import torch
    from torch.utils.data import TensorDataset
    source=TensorDataset(torch.ones(3,2))
    for options in (dict(num_workers=0),dict(num_workers=2,pin_memory=True),
                    dict(num_workers=2,multiprocessing_context='fork')):
        with pytest.raises(ValueError):WorkerTaskDataLoader(source,**options)
