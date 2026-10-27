"""Private PyTorch 2.5.1 implementation, loaded only for task tracing."""
import time
import uuid

from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter
from ._loader_task_trace import _Task, _Batch, _QueueProbe


class _TaskIterator(_MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        self.trace_loader = loader
        super().__init__(loader)

    def _reset(self, loader, first_iter=False):
        if first_iter:
            self._index_queues = [_QueueProbe(q, loader, i)
                                  for i, q in enumerate(self._index_queues)]
        super()._reset(loader, first_iter)

    def _next_index(self):
        indices = super()._next_index()
        loader = self.trace_loader
        return _Task(loader.execution_id, loader.trace_epoch, self._send_idx + 1,
                     uuid.uuid4().hex, indices)

    def _get_data(self):
        index, data = super()._get_data()
        if type(data) is _Batch:
            self.trace_loader._record("receive", dict(receipt=data.receipt,
                                                       at_ns=time.perf_counter_ns()))
        return index, data

    def _next_data(self):
        # This is PyTorch's actual ordered-result buffer, not Queue.qsize().
        ready = len(self._task_info.get(self._rcvd_idx, ())) == 2
        demand = time.perf_counter_ns()
        batch = super()._next_data()
        if type(batch) is not _Batch:
            raise RuntimeError("missing worker result identity")
        self.trace_loader._record("deliver", dict(receipt=batch.receipt,
            demand_ns=demand, at_ns=time.perf_counter_ns(), ordered_buffer_hit=ready,
            observation=None))
        return batch.value
