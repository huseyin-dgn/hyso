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


def test_unsupported_loader_scopes_rejected():
    import torch
    from torch.utils.data import TensorDataset
    source=TensorDataset(torch.ones(3,2))
    for options in (dict(num_workers=0),dict(num_workers=2,pin_memory=True),
                    dict(num_workers=2,multiprocessing_context='fork')):
        with pytest.raises(ValueError):WorkerTaskDataLoader(source,**options)
