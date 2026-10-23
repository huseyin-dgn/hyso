"""Native independent acceptance of the opt-in task adapter."""
import argparse
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
from p21_worker_acceptance import Samples
from trainer_replay_support import seed, snapshot, equal
from hyso.training import Trainer
from hyso.performance import WorkerTaskDataLoader, DataLoaderAnalysisResult, measure_training_with_loader, analyze_worker_tasks
from hyso.performance._loader_event_channel import WorkerEventChannel
from hyso.performance._loader_worker_lifecycle import shutdown_traced_workers


def stochastic_collate(values):
    result = default_collate(values)
    result[0].add_(torch.rand(1).item() + random.random() + np.random.random())
    time.sleep(.002)
    return result


def broken_collate(values):
    raise RuntimeError('intentional collate failure')


def run_failure(case):
    channel=WorkerEventChannel(max_queue_size=32)
    ready=mp.get_context('spawn').Event()
    loader=WorkerTaskDataLoader(Samples(True,case,ready), channel=channel, batch_size=2,
        num_workers=2, persistent_workers=True, timeout=45,
        collate_fn=broken_collate if case=='collate_exception' else stochastic_collate)
    if case=='hang':
        loader.begin_trace_run('explicit-negative-run')
        loader.begin_trace_epoch(1,'training')
        iterator=iter(loader)
        assert ready.wait(45)
    else:
        model=torch.nn.Linear(3,2)
        trainer=Trainer(model,optimizer=torch.optim.SGD(model.parameters(),lr=.01),
                        loss_fn=torch.nn.CrossEntropyLoss(),device='cpu')
        try:
            measure_training_with_loader(trainer,loader,epochs=1)
        except RuntimeError as error:
            assert ('intentional collate failure' in str(error) if case=='collate_exception'
                    else 'exited unexpectedly' in str(error))
        else:
            raise AssertionError('failed task produced training result')
        iterator=loader.trace_iterator
    start=time.monotonic()
    try:
        shutdown_traced_workers(iterator,timeout=2 if case=='hang' else 8)
    except RuntimeError as error:
        if case=='hang':assert 'forced=True' in str(error)
    elapsed=time.monotonic()-start
    workers=iterator._workers
    assert not any(w.is_alive() for w in workers)
    assert loader.trace_source is None
    if case=='hang':assert elapsed<3
    packets=channel.receive_available()
    if case=='collate_exception':assert any(p['collate']['outcome']=='failed' for p in packets)
    channel.close()
    return dict(case=case,cleanup_s=elapsed,exits=[w.exitcode for w in workers],
                packets=len(packets),source_result_created=False)


def run(case):
    torch.set_num_threads(1)
    if case in ('crash','hang','collate_exception'):
        return run_failure(case)
    persistent = case != 'nonpersistent'
    epochs = 2
    outputs = []
    channel = WorkerEventChannel(max_queue_size=1 if case=='saturation' else 64, max_packet_bytes=65536)
    for observed in (False, True):
        seed(97)
        model = torch.nn.Sequential(torch.nn.Linear(3,4), torch.nn.Dropout(.2), torch.nn.Linear(4,2))
        trainer = Trainer(model, optimizer=torch.optim.AdamW(model.parameters(), lr=.01),
                          loss_fn=torch.nn.CrossEntropyLoss(), device='cpu')
        options = dict(batch_size=2, num_workers=2, multiprocessing_context='spawn',
            persistent_workers=persistent, shuffle=True, timeout=45,
            collate_fn=stochastic_collate, generator=torch.Generator().manual_seed(199))
        loader = (WorkerTaskDataLoader(Samples(True), channel=channel, **options) if observed
                  else DataLoader(Samples(), **options))
        order, calls = [], []
        hook=model.register_forward_pre_hook(lambda m,a:order.append(a[0].detach().clone()))
        native=trainer.fit
        def counted(*a, **kw):
            calls.append(1)
            return native(*a, **kw)
        trainer.fit=counted
        try:
            if observed:
                result=measure_training_with_loader(trainer, loader, epochs=epochs)
            else:
                trainer.fit(loader, epochs=epochs)
        finally:
            hook.remove()
        assert len(calls)==1
        outputs.append((snapshot(trainer), loader.generator.get_state(), order))
        iterator=loader.trace_iterator if observed else loader._iterator
        if iterator is not None:
            states=shutdown_traced_workers(iterator, timeout=8)
    equal(outputs[0],outputs[1])
    foreign=result.loader.to_dict()
    foreign['provenance']['loaders']['training']['collate_fn_name']='foreign-source'
    try:
        analyze_worker_tasks(DataLoaderAnalysisResult(foreign),loader)
    except ValueError as error:
        assert 'binding mismatch' in str(error)
    else:
        raise AssertionError('foreign source result consumed task evidence')
    artifact=analyze_worker_tasks(result.loader, loader)
    data=artifact.to_dict()
    if case=='saturation':
        assert data['analysis']['status']=='incomplete'
        assert data['trace_counters']==dict(attempted=8,accepted=1,rejected=7)
        assert data['analysis']['dispatch']['mean_ms'] is None
        channel.close()
        return data
    assert data['analysis']['status']=='complete', data['analysis']
    assert data['analysis']['association']==dict(execution='verified',epoch='verified',batch='verified')
    assert data['analysis']['task_count']==4*epochs
    assert data['analysis']['sample_count']==8*epochs
    assert data['analysis']['stage_timings']['collate']['count']==4*epochs
    assert data['analysis']['stage_timings']['collate']['mean_ms']>=2
    assert data['analysis']['queue_occupancy']['value'] is None
    assert data['analysis']['prefetch_efficiency']['value'] is None
    assert data['analysis']['clock_association']['cross_process_durations']=='unsupported_no_accuracy_bound'
    assert all(r['uncertainty_width_ns']>=0 for r in data['analysis']['clock_association']['bounds'])
    assert result.loader.to_dict()['source_p19']==result.timing.to_dict()
    assert result.loader.to_dict()['source_p20']==result.stages.to_dict()
    assert result.throughput.to_dict()['source_p19']==result.timing.to_dict()
    again=analyze_worker_tasks(result.loader,loader)
    assert again.to_dict()['analysis']['status']=='incomplete'
    try:
        measure_training_with_loader(trainer,loader,epochs=epochs)
    except RuntimeError as error:
        assert 'already belongs' in str(error)
    else:
        raise AssertionError('reused execution accepted')
    channel.close()
    return data


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--admit',required=True)
    args=parser.parse_args()
    deadline=time.monotonic()+10
    while not Path(args.admit).exists():
        if time.monotonic()>deadline:raise TimeoutError('missing Job admission')
        time.sleep(.01)
    data=run(args.case)
    Path(args.output).write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('PASS task adapter',args.case,flush=True)


if __name__=='__main__':
    mp.freeze_support()
    main()
