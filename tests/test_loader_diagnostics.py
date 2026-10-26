"""P21 explicit load stages, safe offline comparisons and bounded native replay."""
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from hyso.training import Trainer
from hyso.performance import (TracedDataset, trace_data_stage, measure_training_with_loader,
    DataLoaderAnalysisResult, WorkerTaskAnalysisResult, WorkerTraceAnalysisResult)
from hyso.performance._loader_worker_trace import WorkerTimingRecorder
from hyso.performance._loader_trace_groups import validate_trace_groups
from hyso.performance.dataloader import _hash


def test_stage_passthrough_exactly_once_and_rng():
    calls=[]
    def operation(value):
        calls.append(value)
        return random.random(), np.random.random(), torch.rand(()), value
    random.seed(1);np.random.seed(2);torch.manual_seed(3)
    expected=operation('same')
    random.seed(1);np.random.seed(2);torch.manual_seed(3)
    actual=trace_data_stage('load',operation,'same')
    assert actual[:2] == expected[:2] and torch.equal(actual[2],expected[2]) and actual[3] == expected[3]
    assert calls == ['same','same']
    with pytest.raises(ValueError):trace_data_stage('invented',operation,'bad')
    assert calls == ['same','same']
    def fail(value):raise RuntimeError('original exception')
    with pytest.raises(RuntimeError,match='original exception'):trace_data_stage('load',fail,None)


def test_explicit_load_group_and_failed_stage():
    recorder=WorkerTimingRecorder(worker_id=0)
    class Data(Dataset):
        def __len__(self):return 1
        def __getitem__(self,index):
            return trace_data_stage('load',lambda x:x+1,index,sample_id=str(index))
    assert TracedDataset(Data(),recorder=recorder)[0] == 1
    events=[dict(e.to_dict(),trace_id='synthetic') for e in recorder.events]
    assert validate_trace_groups(events,expected_stages={'dataset_fetch':1,'load':1})['status']=='complete'
    broken=copy.deepcopy(events);broken[0]['outcome']='failed'
    assert validate_trace_groups(broken)['status']=='incomplete'
    # Existing strict measure_stage behavior is unchanged; opt-in wrapper alone passes through.
    from hyso.performance._loader_stage_trace import measure_stage
    with pytest.raises(RuntimeError):measure_stage(None,'load',lambda x:x,None)


def parent(stages=True):
    torch.manual_seed(27)
    model=torch.nn.Linear(3,2)
    trainer=Trainer(model,optimizer=torch.optim.SGD(model.parameters(),lr=.01),
                    loss_fn=torch.nn.CrossEntropyLoss(),device='cpu')
    loader=DataLoader(TensorDataset(torch.ones(5,3),torch.tensor([0,1,0,1,0])),batch_size=2)
    return measure_training_with_loader(trainer,loader,epochs=1,stages=stages).loader


def test_parent_diagnostics_no_io_no_cuda_and_preserve_exports(tmp_path,monkeypatch):
    result=parent()
    before=result.to_dict()
    def forbidden(*a,**kw):raise AssertionError('analysis must not execute a loader/GPU/process')
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    monkeypatch.setattr(DataLoader,'__iter__',forbidden)
    monkeypatch.setattr(subprocess,'run',forbidden)
    report=result.diagnostics()
    assert len(report['batch_timeline']) == 3
    wait=sum(r['host_ms'] for r in before['observations'] if r['kind']=='batch')
    expected=sum(link['compute'][stage]['host_ms'] for link in before['stage_links'] for stage in ['forward','backward'])
    assert report['findings'][0]['parent_delivered_next_wall_ms'] == wait
    assert report['findings'][0]['cpu_forward_backward_wall_ms'] == pytest.approx(expected)
    assert report['findings'][0]['input_preparation_host_wall_ms'] == pytest.approx(
        sum(r['input_preparation']['host_ms'] for r in before['stage_links']))
    assert report['findings'][0]['samples_per_second'] > 0
    assert [r['input_preparation']['host_wall_ms'] for r in report['batch_timeline']] == [
        r['input_preparation']['host_ms'] for r in before['stage_links']]
    assert report['worker']['status']=='unsupported' and report['worker']['stages']['load']['mean_ms'] is None
    assert report['queue_task_scopes']['physical_queue_residence']['value'] is None
    assert report['recommendations']==[] and result.to_dict()==before
    result.to_json(tmp_path/'parent.json');result.to_csv(tmp_path/'parent.csv')
    assert json.loads((tmp_path/'parent.json').read_text())==before
    report['batch_timeline'].clear()
    assert len(result.diagnostics()['batch_timeline'])==3


def test_no_stage_evidence_never_claims_compute_time():
    report=parent(stages=False).diagnostics()
    assert report['findings'][0]['cpu_forward_backward_wall_ms'] is None
    assert report['findings'][0]['relation']=='no_comparable_CPU_forward_backward_scope'
    assert all(row['compute']['backward']['host_wall_ms'] is None for row in report['batch_timeline'])
    assert all(row['input_preparation']['host_wall_ms'] is None for row in report['batch_timeline'])


@pytest.mark.parametrize('legacy',[False,True])
def test_v1_worker_companion_never_invents_batch_association(legacy):
    # Explicit protocol fixture, not native worker acceptance evidence.
    from hyso.performance import analyze_worker_traces
    raw=parent().to_dict()
    raw['provenance']['loaders']['training'].update(
        dataset_type='hyso.performance._loader_dataset_trace.TracedDataset',num_workers=1,prefetch_factor=2)
    result=DataLoaderAnalysisResult(raw)
    packets=[]
    for index in range(5):
        event=dict(trace_id=str(index),worker_id=0,worker_pid=123,sample_id=str(index),stage='dataset_fetch',
            start_ns=10,end_ns=20,duration_ns=10,duration_ms=.00001,outcome='completed',
            clock='perf_counter_ns',clock_scope='worker_local')
        packets.append(dict(schema_version=1,trace_id=str(index),worker_id=0,worker_pid=123,
            sample_id=str(index),event_count=1,dropped_events=0,events=[event]))
    task=analyze_worker_traces(result,packets,trace_counters=dict(attempted=5,accepted=5,rejected=0),
        worker_states=[dict(worker_id=0,pid=123,alive=False,exitcode=0)]).to_dict()
    if legacy:
        task.pop('evidence_sha256')
        task['analysis'].pop('association')
    companion=WorkerTraceAnalysisResult(task)
    report=result.diagnostics(worker_traces=companion)
    assert report['worker']['status']=='complete'
    assert report['worker']['association']==dict(execution='unknown',epoch='unknown',batch='unknown')
    assert all(row['worker_task'] is None for row in report['batch_timeline'])
    assert report['worker']['stages']['load']['mean_ms'] is None
    assert report['worker']['stages']['dataset_fetch']['count']==5
    assert companion.to_dict()==task
    foreign=result.to_dict();foreign['provenance']['loaders']['training']['dataset_type']='other'
    with pytest.raises(ValueError,match='different P21 source'):
        DataLoaderAnalysisResult(foreign).diagnostics(worker_traces=companion)
    with pytest.raises(ValueError,match='choose one'):
        result.diagnostics(worker_traces=companion,worker_tasks=object())


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    if os.name!='nt':pytest.skip('Windows spawn acceptance')
    folder=tmp_path_factory.mktemp('p21-diagnostics')
    tests=Path(__file__).resolve().parent
    cache={}
    def execute(case):
        if case not in cache:
            path=folder/(case+'.json')
            outcome=subprocess.run([sys.executable,str(tests/'p21_native_supervisor.py'),
                '--script',str(tests/'p21_diagnostics_acceptance.py'),'--case',case,'--output',str(path),
                '--timeout','150'],capture_output=True,text=True,timeout=170)
            logs='\n'.join(p.read_text(encoding='utf-8',errors='replace') for p in folder.glob(case+'.*.log'))
            assert outcome.returncode==0,outcome.stdout+outcome.stderr+logs
            receipt=json.loads(path.with_suffix('.receipt.json').read_text())
            assert receipt['exit_code']==0 and receipt['timed_out'] is False
            cache[case]=json.loads(path.read_text())
        return copy.deepcopy(cache[case])
    return execute


@pytest.mark.parametrize('case',['standard_images','task_images'])
def test_real_file_spawn_integrity(native,case):
    data=native(case)
    assert data['exact_batch_model_optimizer_rng_generator_history']
    assert data['all_workers_exited'] and data['real_files']
    assert data['diagnostics']['source_p21_sha256']==_hash(data['parent'])


@pytest.mark.parametrize('case,workers,prefetch,pin',[
    ('standard_images_inline',0,None,False),
    ('standard_images_single_prefetch_pin',1,1,True),
])
def test_real_file_loader_configuration_integrity(native,case,workers,prefetch,pin):
    data=native(case)
    test_real_file_spawn_integrity(native,case)
    result=DataLoaderAnalysisResult(data['parent'])
    report=result.diagnostics()
    config=report['core']['configurations']['training']
    assert (config['num_workers'],config['prefetch_factor'],config['pin_memory'])==(workers,prefetch,pin)
    assert config['persistent_workers'] is (workers > 0)
    assert data['batch_tensor_is_pinned']==[[False]*6]*2
    assert report['worker']['status']=='unsupported'
    assert all(row['worker_task'] is None for row in report['batch_timeline'])
    assert len(report['batch_timeline'])==6
    assert [row['samples'] for row in report['batch_timeline']]==[2,3,3]*2
    assert all(row['cpu_forward_backward_wall_ms'] is not None for row in report['batch_timeline'])
    assert all(row['samples_per_second'] > 0 for row in report['findings'])
    assert report['recommendations']==[]


def test_task_link_scope_and_incomplete_or_foreign_evidence(native):
    data=native('task_images')
    result=DataLoaderAnalysisResult(data['parent'])
    task=WorkerTaskAnalysisResult(data['task_evidence'])
    report=result.diagnostics(worker_tasks=task)
    assert report['worker']['status']=='complete'
    assert all(row['worker_task']['worker_service']['clock_scope']=='worker_local' for row in report['batch_timeline'])
    assert all(row['worker_task']['parent_buffer']['clock_scope']=='parent_local' for row in report['batch_timeline'])
    assert report['queue_task_scopes']['parent_index_queue_put']['status']=='measured'
    assert report['queue_task_scopes']['task_lifecycle']['status']=='measured'
    foreign=copy.deepcopy(data['parent']);foreign['provenance']['loaders']['training']['dataset_type']='other'
    with pytest.raises(ValueError,match='different P21 source'):
        DataLoaderAnalysisResult(foreign).diagnostics(worker_tasks=task)
    from hyso.performance.worker_task_result import _derive
    broken=task.to_dict();broken['packets'].pop()
    broken['analysis']=_derive(broken)
    broken['evidence_sha256']=_hash({k:v for k,v in broken.items() if k not in ['analysis','evidence_sha256']})
    report=result.diagnostics(worker_tasks=WorkerTaskAnalysisResult(broken))
    assert report['worker']['status']=='incomplete'
    assert report['worker']['service']['mean_ms'] is None
    assert report['queue_task_scopes']['parent_index_queue_put']['mean_ms'] is None
    assert all(row['worker_task'] is None for row in report['batch_timeline'])


def test_task_clock_offsets_never_change_same_clock_durations(native):
    data=native('task_images');raw=data['task_evidence']
    from hyso.performance.worker_task_result import _derive
    before=DataLoaderAnalysisResult(data['parent']).diagnostics(worker_tasks=WorkerTaskAnalysisResult(raw))
    # Independent offsets per worker; parent timestamps remain unchanged.
    for packet in raw['packets']:
        offset=(packet['receipt']['worker_id']+1)*10**15
        for key in ['receive_ns','finish_ns']:packet['receipt'][key]+=offset
        for key in ['start_ns','end_ns']:packet['collate'][key]+=offset
        for sample in packet['samples']:
            for event in sample['events']:
                for key in ['start_ns','end_ns']:event[key]+=offset
    for row in raw['ledger']:
        if row['kind'] in ['receive','deliver']:
            offset=(row['receipt']['worker_id']+1)*10**15
            for key in ['receive_ns','finish_ns']:row['receipt'][key]+=offset
    raw['analysis']=_derive(raw)
    raw['evidence_sha256']=_hash({k:v for k,v in raw.items() if k not in ['analysis','evidence_sha256']})
    after=DataLoaderAnalysisResult(data['parent']).diagnostics(worker_tasks=WorkerTaskAnalysisResult(raw))
    assert after['worker']['status']=='complete'
    assert after['batch_timeline']==before['batch_timeline']
    assert after['findings']==before['findings']
