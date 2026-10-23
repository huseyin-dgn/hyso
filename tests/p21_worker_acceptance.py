"""Isolated native P21 acceptance cases, admitted by the Windows Job supervisor."""
import argparse
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch
from torch.utils.data import DataLoader, Dataset
from hyso.performance import measure_training_with_loader
from hyso.performance._loader_dataset_trace import TracedDataset
from hyso.performance._loader_event_channel import WorkerEventChannel, audit_trace_delivery
from hyso.performance._loader_stage_trace import measure_stage
from hyso.performance._loader_worker_lifecycle import shutdown_traced_workers
from hyso.performance.worker_trace_result import analyze_worker_traces
from trainer_replay_support import seed, snapshot, equal


class Samples(Dataset):
    def __init__(self, traced=False, failure=None, ready=None):
        self.traced, self.failure = traced, failure
        self.ready = ready

    def __len__(self):
        return 8

    def __getitem__(self, index):
        if self.failure == 'hang':
            self.ready.set()
            time.sleep(60)
        if index == 2 and self.failure == 'exception':
            raise RuntimeError('intentional dataset failure')
        if index == 2 and self.failure == 'crash':
            os._exit(7)
        value = torch.tensor([index / 20, .1, .2])
        for _ in range(index % 3):
            if self.traced:
                value = measure_stage(None, 'decode', lambda x: x + .01, value, sample_id=str(index))
            else:
                value = value + .01
        if self.traced:
            value = measure_stage(None, 'transform', lambda x: x * 2, value, sample_id=str(index))
        else:
            value = value * 2
        return value, index % 2


def make_loader(dataset):
    return DataLoader(dataset, batch_size=2, num_workers=2, multiprocessing_context='spawn',
                      persistent_workers=True, shuffle=True,
                      generator=torch.Generator().manual_seed(192), timeout=45)


def exits(states):
    return {s['worker_id']: s['exitcode'] for s in states}


def lock_die(channel, ready):
    channel._lock.acquire()
    ready.set()
    os._exit(7)


def run(case):
    torch.set_num_threads(1)
    if case == 'lock_crash':
        channel = WorkerEventChannel(max_queue_size=2, lock_timeout=.1)
        ctx = mp.get_context('spawn')
        ready = ctx.Event()
        child = ctx.Process(target=lock_die, args=(channel, ready))
        child.start()
        assert ready.wait(45)
        child.join(5)
        assert child.exitcode == 7
        start = time.monotonic()
        for operation in (channel.trace_counters, channel.receive_available,
                          lambda: channel.send_trace({})):
            try:
                operation()
            except TimeoutError:
                pass
            else:
                raise AssertionError('abandoned lock reported success')
        channel.close()
        assert time.monotonic() - start < 1
        return dict(case=case, exitcode=child.exitcode, elapsed=time.monotonic()-start)

    if case == 'shutdown_timeout':
        import threading
        class Stuck:
            _workers = []
            def _shutdown_workers(self):
                threading.Event().wait(30)
        start = time.monotonic()
        try:
            shutdown_traced_workers(Stuck(), timeout=.3)
        except RuntimeError as error:
            assert 'forced=True' in str(error)
        else:
            raise AssertionError('timeout reported clean')
        assert time.monotonic() - start < .7
        return dict(case=case, elapsed=time.monotonic()-start)

    channel = WorkerEventChannel(max_queue_size=1 if case == 'saturation' else 64)
    ready = mp.get_context('spawn').Event() if case == 'hang' else None
    dataset = TracedDataset(Samples(True, case, ready), channel=channel, trace_batches=True,
                            max_events_per_sample=1 if case == 'overflow' else 64)
    loader = make_loader(dataset)
    if case == 'hang':
        iterator = iter(loader)
        assert ready.wait(45), 'worker never entered the hanging dataset'
        workers = list(iterator._workers)
        start = time.monotonic()
        try:
            shutdown_traced_workers(iterator, timeout=2)
        except RuntimeError as error:
            assert 'forced=True' in str(error)
        else:
            raise AssertionError('hanging worker reported clean shutdown')
        elapsed = time.monotonic() - start
        assert elapsed < 3, elapsed
        assert not any(w.is_alive() for w in workers)
        channel.close()
        return dict(case=case, cleanup_s=elapsed, exits=[w.exitcode for w in workers])
    if case in ('exception', 'crash'):
        try:
            list(loader)
        except RuntimeError as error:
            assert ('intentional dataset failure' in str(error) if case == 'exception' else 'exited unexpectedly' in str(error))
        else:
            raise AssertionError('worker failure was hidden')
        workers = list(loader._iterator._workers)
        start = time.monotonic()
        try:
            states = shutdown_traced_workers(loader._iterator, timeout=4)
        except RuntimeError:
            states = [dict(worker_id=i, pid=w.pid, alive=w.is_alive(), exitcode=w.exitcode)
                      for i,w in enumerate(workers)]
        assert not any(s['alive'] for s in states), states
        packets = channel.receive_available()
        audit = audit_trace_delivery(packets, expected_fetches=8,
                    trace_counters=channel.trace_counters(), worker_exit_codes=exits(states))
        assert audit['status'] == 'incomplete', audit
        channel.close()
        return dict(case=case, states=states, audit=audit, cleanup_s=time.monotonic()-start)

    outputs = []
    if case == 'training':
        from hyso.training import Trainer
        for observed in (False, True):
            seed(81)
            model = torch.nn.Sequential(torch.nn.Linear(3,4), torch.nn.Dropout(.2), torch.nn.Linear(4,2))
            trainer = Trainer(model, optimizer=torch.optim.AdamW(model.parameters(), lr=.01),
                              loss_fn=torch.nn.CrossEntropyLoss(), device='cpu')
            current = loader if observed else make_loader(Samples())
            order = []
            hook = model.register_forward_pre_hook(lambda m, args: order.append(args[0].detach().clone()))
            calls = []
            native = trainer.fit
            def counted(*a, **kw):
                calls.append(1)
                return native(*a, **kw)
            trainer.fit = counted
            try:
                if observed:
                    result = measure_training_with_loader(trainer, current, epochs=2, stages=True)
                else:
                    trainer.fit(current, epochs=2)
            finally:
                hook.remove()
            outputs.append((snapshot(trainer), current.generator.get_state(), order))
            assert len(calls) == 1
            states = shutdown_traced_workers(current._iterator, timeout=8)
        equal(outputs[0], outputs[1])
        parent = result.loader.to_dict()
        assert parent['source_p19'] == result.timing.to_dict()
        assert parent['source_p20'] == result.stages.to_dict()
        assert result.throughput.to_dict()['source_p19'] == result.timing.to_dict()
    else:
        values = [batch[0] for _ in range(2) for batch in loader]
        assert len(values) == 8
        states = shutdown_traced_workers(loader._iterator, timeout=8)
    counters = channel.trace_counters()
    packets = channel.receive_available()
    audit = audit_trace_delivery(packets, expected_fetches=16, trace_counters=counters,
                                worker_exit_codes=exits(states))
    assert counters['attempted'] == 16
    assert counters['attempted'] == counters['accepted'] + counters['rejected']
    assert len(packets) == counters['accepted']
    assert audit['status'] == ('complete' if case == 'training' else 'incomplete'), audit
    if case == 'saturation':
        assert counters == dict(attempted=16, accepted=1, rejected=15)
    if case == 'overflow':
        assert audit['recorder_dropped_events'] > 0
        assert sum(p['dropped_events'] for p in packets) == 30
    if case == 'training':
        assert len({p['trace_id'] for p in packets}) == 16
        assert {p['worker_id'] for p in packets} == {0,1}
        assert all(sum(p['sample_id'] == str(i) for p in packets) == 2 for i in range(8))
        assert {p['event_count'] for p in packets} == {2,3,4}
        artifact = analyze_worker_traces(result.loader, packets, trace_counters=counters, worker_states=states)
        data = artifact.to_dict()
        assert data['analysis']['status'] == 'complete', data['analysis']
        assert data['analysis']['association'] == dict(execution='unknown', epoch='unknown', batch='unknown')
        assert result.loader.to_dict() == parent
        assert parent['status'] == 'partial'
        assert parent['worker_service']['value'] is None
        channel.close()
        return data
    channel.close()
    return dict(case=case, audit=audit, states=states, counters=counters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--admit', required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + 10
    while not Path(args.admit).exists():
        if time.monotonic() > deadline:
            raise TimeoutError('supervisor admission missing')
        time.sleep(.01)
    data = run(args.case)
    Path(args.output).write_text(json.dumps(data, indent=2), encoding='utf-8')
    print('PASS', args.case, flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()
