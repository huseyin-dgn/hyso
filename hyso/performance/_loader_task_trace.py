"""Opt-in, version-gated PyTorch task receipts; legacy tracing is unchanged.

Only parent iterator methods are extended. PyTorch's worker loop, queues,
sampler, seeding, fetch scheduling and shutdown implementation are retained.
"""
from dataclasses import dataclass
import hashlib
import inspect
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info, default_collate

from ._loader_dataset_trace import TracedDataset
from ._loader_worker_trace import WorkerTimingRecorder
from ._loader_event_channel import WorkerEventChannel


ADAPTER = "torch-2.5.1-map-tasks-v1"
SOURCE_HASH = "e59741680fd0aeaba408f39f5b560bca47d3c3759fabd6d6c428e4f165e5acf3"


def check_adapter():
    # Private APIs are required only by the explicit, version-gated adapter.
    if torch.__version__.split("+")[0] != "2.5.1":
        raise RuntimeError("unsupported PyTorch task adapter: exact 2.5.1 source required")
    try:
        from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
        digest = hashlib.sha256("".join(inspect.getsource(getattr(
            _MultiProcessingDataLoaderIter, n)) for n in
            ("_reset", "_try_put_index", "_next_data", "_get_data")).encode()).hexdigest()
    except (ImportError, AttributeError, OSError, TypeError) as exc:
        raise RuntimeError("unsupported PyTorch task adapter: source unavailable") from exc
    if digest != SOURCE_HASH:
        raise RuntimeError("unsupported PyTorch task adapter: exact 2.5.1 source required")


def clock_info():
    info = time.get_clock_info("perf_counter")
    return dict(implementation=info.implementation, monotonic=info.monotonic,
                resolution_ns=info.resolution * 1e9)


@dataclass(frozen=True)
class _Task:
    execution_id: str
    epoch_id: int
    batch_id: int
    token: str
    indices: list


@dataclass
class _Fetched:
    task: _Task
    values: list
    samples: list
    receive_ns: int


@dataclass
class _Batch:
    value: object
    receipt: dict


class _TaskDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitems__(self, task):
        if type(task) is not _Task:
            raise RuntimeError("missing explicit worker task identity")
        receive = time.perf_counter_ns()
        values, samples = [], []
        for position, index in enumerate(task.indices):
            recorder = WorkerTimingRecorder(max_events=1000)
            values.append(TracedDataset(self.dataset, recorder=recorder)[index])
            events = [event.to_dict() for event in recorder.events]
            trace_id = task.token + ":" + str(position)
            for event in events:
                event["trace_id"] = trace_id
            samples.append(dict(position=position, trace_id=trace_id,
                                sample_id=str(index), dropped_events=recorder.dropped_events,
                                events=events))
        return _Fetched(task, values, samples, receive)

    def __getitem__(self, index):
        raise RuntimeError("task tracing requires automatic batching")


class _TaskCollate:
    def __init__(self, operation, channel):
        self.operation, self.channel = operation, channel

    def __call__(self, fetched):
        task = fetched.task
        info = get_worker_info()
        receipt = dict(execution_id=task.execution_id, epoch_id=task.epoch_id,
                       batch_id=task.batch_id, token=task.token, worker_id=info.id,
                       worker_pid=os.getpid(), receive_ns=fetched.receive_ns)
        start = time.perf_counter_ns()
        outcome = "completed"
        try:
            value = self.operation(fetched.values)
        except BaseException:
            outcome = "failed"
            raise
        finally:
            end = time.perf_counter_ns()
            receipt["finish_ns"] = end
            packet = dict(schema_version=2, receipt=dict(receipt), samples=fetched.samples,
                          collate=dict(start_ns=start, end_ns=end, duration_ns=end-start,
                                       outcome=outcome), clock=clock_info())
            try:
                self.channel.send_trace(packet)
            except (OSError, RuntimeError, ValueError):
                # Training data/exceptions retain their original semantics.
                # Missing packets/counters fail the final conservation audit.
                pass
        return _Batch(value, receipt)


class _QueueProbe:
    """Parent-only proxy around the actual index_queue.put invocation."""
    def __init__(self, queue, loader, worker):
        self.queue, self.loader, self.worker = queue, loader, worker

    def put(self, item, *args, **kwargs):
        if not (type(item) is tuple and len(item) == 2 and type(item[1]) is _Task):
            return self.queue.put(item, *args, **kwargs)
        task = item[1]
        start = time.perf_counter_ns()
        result = self.queue.put(item, *args, **kwargs)
        end = time.perf_counter_ns()
        self.loader._record("submit", dict(execution_id=task.execution_id,
            epoch_id=task.epoch_id, batch_id=task.batch_id, token=task.token,
            indices=[str(i) for i in task.indices], worker_id=self.worker,
            start_ns=start, end_ns=end, duration_ns=end-start))
        return result

    def __getattr__(self, name):
        return getattr(self.queue, name)


class WorkerTaskDataLoader(DataLoader):
    """Explicit task/collate tracing for existing map-style dataset operations.

    Supports automatic batching, CPU output, spawn, persistent/nonpersistent
    workers and custom samplers/collate functions. One measurement run per loader.
    Bulk __getitems__, IterableDataset, pin_memory and arbitrary torch versions
    are rejected rather than silently changing their behavior.
    """
    def __init__(self, dataset, *, channel=None, **kwargs):
        check_adapter()
        if not isinstance(dataset, Dataset) or isinstance(dataset, IterableDataset):
            raise TypeError("task tracing requires a map-style Dataset")
        if callable(getattr(dataset, "__getitems__", None)):
            raise ValueError("bulk __getitems__ task tracing is unsupported")
        if kwargs.get("num_workers", 0) < 1 or kwargs.get("pin_memory", False):
            raise ValueError("task tracing requires workers > 0 and pin_memory=False")
        if kwargs.get("multiprocessing_context", "spawn") != "spawn":
            raise ValueError("task adapter currently supports explicit spawn only")
        kwargs["multiprocessing_context"] = "spawn"
        operation = kwargs.pop("collate_fn", None) or default_collate
        self.trace_channel = WorkerEventChannel() if channel is None else channel
        self.execution_id = None
        self.trace_epoch = None
        self.trace_ledger = []
        self.trace_source = None
        self.trace_processes = []
        self.trace_budget = 100000
        super().__init__(_TaskDataset(dataset), collate_fn=_TaskCollate(operation, self.trace_channel), **kwargs)
        if not self._auto_collation:
            raise ValueError("task tracing requires automatic batching")

    def begin_trace_run(self, execution_id):
        if self.execution_id is not None:
            raise RuntimeError("task loader already belongs to a measurement run")
        self.execution_id = execution_id
        if self.trace_channel.trace_counters() != dict(attempted=0, accepted=0, rejected=0):
            raise RuntimeError("task trace channel must be fresh")

    def begin_trace_epoch(self, epoch_id, phase):
        if phase != "training" or self.execution_id is None:
            raise RuntimeError("task adapter requires explicit training run/epoch")
        self.trace_epoch = epoch_id

    def _record(self, kind, payload):
        if len(self.trace_ledger) >= self.trace_budget:
            raise RuntimeError("task receipt budget exceeded")
        self.trace_ledger.append(dict(sequence=len(self.trace_ledger), kind=kind, **payload))

    def bind_trace_delivery(self, observation):
        last = self.trace_ledger[-1]
        if last["kind"] != "deliver" or last["observation"] is not None:
            raise RuntimeError("unmatched parent delivery")
        last["observation"] = observation

    def _get_iterator(self):
        if self.trace_epoch is None:
            raise RuntimeError("use measure_training_with_loader for explicit epoch identity")
        from ._loader_task_iterator import _TaskIterator
        iterator = _TaskIterator(self)
        self.trace_iterator = iterator
        self.trace_processes.append((self.trace_epoch, list(iterator._workers)))
        return iterator
