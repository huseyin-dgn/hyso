"""P15 bounded owned-buffer copies. Event intervals are not CUPTI activities."""
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .cold import _terminate
from .inference import _base
from .result import TransferResult, _integer, _positive, _plain


@dataclass(frozen=True)
class TransferProtocol:
    shape: tuple[int, ...] = (262144,)
    dtype: str = 'float32'
    directions: tuple[str, ...] = ('H2D', 'D2H')
    pinned: bool = False
    non_blocking: bool = False
    device: str = 'cuda:0'
    layout: str = 'contiguous_dense'
    warmup_iterations: int = 2
    measure_iterations: int = 5
    repeats: int = 2
    timeout_seconds: float = 45.
    cpu_memory_limit_bytes: int = 3*1024**3
    input_limit_bytes: int = 8*1024**2
    cuda_memory_fraction: float = .1

    def __post_init__(self):
        if type(self.shape) not in (tuple, list) or not 1 <= len(self.shape) <= 8:
            raise ValueError('explicit nonempty shape required')
        for n in self.shape: _integer(n, 1)
        object.__setattr__(self, 'shape', tuple(self.shape))
        if type(self.directions) not in (tuple, list) or not self.directions or len(set(self.directions)) != len(self.directions):
            raise ValueError('unique directions required')
        if any(d not in ('H2D', 'D2H') for d in self.directions): raise ValueError('H2D/D2H only')
        object.__setattr__(self, 'directions', tuple(self.directions))
        for v in (self.pinned, self.non_blocking):
            if type(v) is not bool: raise TypeError('explicit bool policy required')
        for v in (self.dtype, self.device, self.layout):
            if type(v) is not str: raise TypeError('string profile required')
        _integer(self.warmup_iterations); _integer(self.measure_iterations, 1); _integer(self.repeats, 1)
        if (self.warmup_iterations+self.measure_iterations)*self.repeats*len(self.directions) > 256:
            raise ValueError('at most 256 copies including warmup')
        _positive(self.timeout_seconds)
        if self.timeout_seconds > 600: raise ValueError('timeout at most 600s')
        _integer(self.cpu_memory_limit_bytes, 256*1024**2); _integer(self.input_limit_bytes, 1)
        if self.input_limit_bytes > 64*1024**2: raise ValueError('input budget at most 64MiB')
        _positive(self.cuda_memory_fraction)
        if self.cuda_memory_fraction > 1: raise ValueError('invalid allocator fraction')


class TransferError(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics = _plain(diagnostics)
        super().__init__(json.dumps(self.diagnostics, sort_keys=True))


def _transfer_metrics(rows):
    directions = []
    for direction in dict.fromkeys(r['direction'] for r in rows):
        selected = [r for r in rows if r['direction'] == direction]
        elapsed = math.fsum(r['device_stream_ms'] for r in selected)
        valid = all(r['terminal_complete'] and r['equal'] for r in selected) and elapsed > 0
        directions.append(dict(direction=direction, completed_copies=len(selected),
            bytes=sum(r['transfer_bytes'] for r in selected), device_stream_ms=elapsed,
            completed_bytes_per_second=sum(r['transfer_bytes'] for r in selected)/(elapsed/1000) if valid else None,
            bandwidth_duration='sum_cuda_event_stream_intervals_ms',
            status='measured' if valid else 'invalid_duration_or_completion'))
    return dict(status='partial', reason='overlap_requires_verified_P14_copy_compute_timeline',
        directions=directions, overlap=dict(value=None, status='unsupported',
            reason='no_verified_common_copy_compute_timeline_or_kernel_correlation'),
        asynchronous_execution=dict(value=None, status='not_proven',
            reason='non_blocking_is_request_not_overlap_or_host_async_proof'))


def measure_transfers(*, protocol=TransferProtocol()):
    """Fresh Windows worker; recipe only, never consumes a caller Tensor.

    Dense contiguous buffers on cuda:0 only. Pinned refers to H2D source or
    D2H destination. Validation reverse copies are outside measured intervals.
    Native exits are crashed, never inferred OOM. Errors carry plain receipts.
    """
    if type(protocol) is not TransferProtocol: raise TypeError('TransferProtocol required')
    receipt = dict(status='failed', reason=None, worker_pid=None, launcher_pid=None,
                   worker_phase=None, exit_code=None, process_terminated=False, stderr_tail='')
    worker = None; start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='hyso-transfer-') as directory:
        root = Path(directory); request=root/'request.json'; output=root/'result.json'; ready=root/'ready.json'
        request.write_text(json.dumps(asdict(protocol)), encoding='utf-8')
        command = [sys.executable, '-I', '-B', str(Path(__file__).with_name('_transfer_worker.py')),
                   str(request), str(output), str(ready), str(Path(__file__).resolve().parents[2])]
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                worker = subprocess.Popen(command, stdout=stdout, stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                receipt['launcher_pid'] = worker.pid
                try: worker.wait(timeout=protocol.timeout_seconds)
                except subprocess.TimeoutExpired:
                    receipt.update(status='timed_out', reason='owned worker deadline'); _terminate(worker)
                    worker.wait(timeout=10)
                receipt['exit_code'] = worker.returncode
                if ready.exists():
                    r=json.loads(ready.read_text(encoding='utf-8'))
                    receipt.update(worker_pid=r['pid'], worker_phase=r['phase'])
                if receipt['status'] != 'timed_out':
                    if worker.returncode != 0:
                        receipt.update(status='crashed', reason='unexpected native/process exit; not OOM')
                    elif output.exists():
                        if output.stat().st_size > 2*1024**2: raise ValueError('result evidence budget')
                        body=_plain(json.loads(output.read_text(encoding='utf-8')))
                        if body['worker_pid'] != receipt['worker_pid']: raise ValueError('worker PID mismatch')
                        receipt.update(status=body['status'], reason=body['reason'])
                    else: receipt['reason']='missing worker result'
            finally:
                if worker is not None:
                    if worker.poll() is None: _terminate(worker)
                    worker.wait(timeout=10); receipt['process_terminated']=worker.poll() is not None
                stderr.seek(0,2); length=stderr.tell(); stderr.seek(max(0,length-4096))
                receipt['stderr_tail']=stderr.read().decode('utf-8',errors='replace')
        receipt['execution_seconds']=time.perf_counter()-start
        if receipt['status'] != 'completed': raise TransferError(receipt)
        data=body['result']; data['raw_evidence']['worker']=receipt
        return TransferResult(data)


def _validate_transfer(data):
    plan=TransferProtocol(**data['protocol']['configuration']); raw=data['raw_evidence']; rows=raw['copies']
    count=len(plan.directions)*plan.repeats*plan.measure_iterations
    if len(rows)!=count or data['actual']!=dict(completed_copies=count,
        warmup_copies=len(plan.directions)*plan.repeats*plan.warmup_iterations,
        validation_copies=count, pending_copies=0): raise ValueError('invalid copy completions')
    if data['requested']!=dict(measured_copies=count): raise ValueError('invalid requested copies')
    tensor=raw['tensor']; _integer(tensor['numel'],1); _integer(tensor['element_size'],1)
    sizes={'float16':2,'float32':4,'float64':8,'int32':4,'int64':8,'uint8':1}
    if tensor!=dict(shape=list(plan.shape),numel=math.prod(plan.shape),element_size=sizes[plan.dtype],
                    dtype=plan.dtype,layout='contiguous_dense',transfer_bytes=math.prod(plan.shape)*sizes[plan.dtype]):
        raise ValueError('invalid actual tensor accounting')
    expected=[(d,r,i) for d in plan.directions for r in range(plan.repeats) for i in range(plan.measure_iterations)]
    for row,key in zip(rows,expected):
        if (row['direction'],row['repeat'],row['iteration'])!=key: raise ValueError('copy order')
        for k in ('host_start_ns','host_end_ns','terminal_end_ns','transfer_bytes'): _integer(row[k])
        if not row['host_start_ns']<=row['host_end_ns']<=row['terminal_end_ns']: raise ValueError('copy clock')
        if row['host_enqueue_ms']!=(row['host_end_ns']-row['host_start_ns'])/1e6: raise ValueError('enqueue mismatch')
        v=row['device_stream_ms']
        if type(v) not in (int,float) or not math.isfinite(v) or v<0: raise ValueError('copy duration')
        if row['transfer_bytes']!=tensor['transfer_bytes'] or row['equal'] is not True or row['terminal_complete'] is not True:
            raise ValueError('missing equality/completion evidence')
        if row['host_pinned'] is not plan.pinned or row['non_blocking_requested'] is not plan.non_blocking:
            raise ValueError('host policy mismatch')
    if data['measurements']!=_transfer_metrics(rows): raise ValueError('copy metrics mismatch')
