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
    """Owned copies; resource quotas are opt-in, overlap probing is separate."""
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
    cpu_memory_limit_bytes: int | None = None
    input_limit_bytes: int | None = None
    cuda_memory_fraction: float | None = None
    probe_overlap: bool = False

    def __post_init__(self):
        if type(self.shape) not in (tuple, list) or not 1 <= len(self.shape) <= 8:
            raise ValueError('explicit nonempty shape required')
        for n in self.shape: _integer(n, 1)
        object.__setattr__(self, 'shape', tuple(self.shape))
        if type(self.directions) not in (tuple, list) or not self.directions or len(set(self.directions)) != len(self.directions):
            raise ValueError('unique directions required')
        if any(d not in ('H2D', 'D2H') for d in self.directions): raise ValueError('H2D/D2H only')
        object.__setattr__(self, 'directions', tuple(self.directions))
        for v in (self.pinned, self.non_blocking, self.probe_overlap):
            if type(v) is not bool: raise TypeError('explicit bool policy required')
        for v in (self.dtype, self.device, self.layout):
            if type(v) is not str: raise TypeError('string profile required')
        _integer(self.warmup_iterations); _integer(self.measure_iterations, 1); _integer(self.repeats, 1)
        if (self.warmup_iterations+self.measure_iterations)*self.repeats*len(self.directions) > 256:
            raise ValueError('at most 256 copies including warmup')
        _positive(self.timeout_seconds)
        if self.timeout_seconds > 600: raise ValueError('timeout at most 600s')
        if self.cpu_memory_limit_bytes is not None: _integer(self.cpu_memory_limit_bytes, 256*1024**2)
        if self.input_limit_bytes is not None: _integer(self.input_limit_bytes, 1)
        if self.cuda_memory_fraction is not None:
            _positive(self.cuda_memory_fraction)
            if self.cuda_memory_fraction > 1: raise ValueError('invalid allocator fraction')


class TransferError(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics = _plain(diagnostics)
        super().__init__(json.dumps(self.diagnostics, sort_keys=True))


def _transfer_metrics(rows, *, evidence_version=1, overlap_probe=None):
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
    result=dict(status='partial', reason='overlap_requires_verified_P14_copy_compute_timeline',
        directions=directions, overlap=dict(value=None, status='unsupported',
            reason='no_verified_common_copy_compute_timeline_or_kernel_correlation'),
        asynchronous_execution=dict(value=None, status='not_proven',
            reason='non_blocking_is_request_not_overlap_or_host_async_proof'))
    if evidence_version==1: return result
    if evidence_version!=2: raise ValueError('unsupported transfer evidence version')
    for summary in directions:
        selected=[r for r in rows if r['direction']==summary['direction']]
        bps=summary['completed_bytes_per_second']
        summary.update(effective_bandwidth_gb_s=bps/1e9 if bps is not None else None,
            bandwidth_unit='GB/s (10^9 bytes/s)',
            host_enqueue_ms=math.fsum(r['host_enqueue_ms'] for r in selected),
            host_completion_wall_ms=math.fsum((r['terminal_end_ns']-r['host_start_ns'])/1e6 for r in selected),
            timing_unit='ms',
            timing_scope='copy_only; preparation_warmup_and_validation_excluded; CUDA_events_include_host_staging_and_scheduling')
    observed=[i for i,r in enumerate(rows) if not r['start_complete_after_enqueue']]
    result.update(reason='hardware_copy_compute_overlap_not_proven',
        transfer_status='measured' if all(r['status']=='measured' for r in directions) else 'partial',
        asynchronous_execution=dict(value=True if observed else None,
            status='observed' if observed else 'unsupported',
            reason=None if observed else 'no_host_return_before_copy_start_event_observed',
            scope='host_copy_call_returned_before_same_stream_start_event_completed; not_hardware_overlap',
            observed_copy_indices=observed),
        overlap=dict(value=None,status='unsupported',
            reason='CUDA_event_windows_do_not_prove_hardware_copy_compute_overlap',
            probe_status='observed' if overlap_probe is not None else 'not_requested',
            stream_window_intersection_ms=max(0.,min(overlap_probe['copy_end_ms'],overlap_probe['compute_end_ms'])-
                max(overlap_probe['copy_start_ms'],overlap_probe['compute_start_ms'])) if overlap_probe is not None else None,
            interval_unit='ms',interval_scope='same_epoch_CUDA_event_windows; not_copy_engine_or_SM_activity'))
    return result


def measure_transfers(*, protocol=TransferProtocol()):
    """Fresh Windows worker; recipe only, never consumes a caller Tensor.

    Dense contiguous buffers on cuda:0 only. Pinned refers to H2D source or
    D2H destination. Validation reverse copies are outside measured intervals.
    Native exits are crashed, never inferred OOM. Errors carry plain receipts.
    probe_overlap adds an unmeasured H2D/float32 matmul stream-window probe;
    hardware overlap remains unsupported without direct activity evidence.
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
    version=data['protocol'].get('evidence_version',1)
    if type(version) is not int or version not in (1,2): raise ValueError('unsupported transfer evidence version')
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
    for index,(row,key) in enumerate(zip(rows,expected)):
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
        if version==2:
            if row['events_preinitialized'] is not True: raise ValueError('copy events not initialized before timing')
            _integer(raw['stream_id'])
            if (row['start_stream_id']!=raw['stream_id'] or row['end_stream_id']!=raw['stream_id']
                    or type(row['start_stream_id']) is not int or type(row['end_stream_id']) is not int):
                raise ValueError('copy event stream mismatch')
            if (type(row['start_event_index']) is not int or row['start_event_index']!=2*index
                    or type(row['end_event_index']) is not int or row['end_event_index']!=2*index+1):
                raise ValueError('copy event pair mismatch')
            for k in ('start_complete_after_enqueue','end_complete_after_enqueue'):
                if type(row[k]) is not bool: raise ValueError('invalid copy completion observation')
            _integer(row['host_observation_ns'])
            if not row['host_end_ns']<=row['host_observation_ns']<=row['terminal_end_ns']:
                raise ValueError('copy observation outside host window')
    probe=raw.get('overlap_probe') if version==2 else None
    if version==2:
        worker=raw['worker']
        if (worker['status']!='completed' or worker['exit_code']!=0 or worker['process_terminated'] is not True):
            raise ValueError('transfer worker did not finish normally')
        if (probe is not None)!=plan.probe_overlap: raise ValueError('overlap probe request mismatch')
        if probe is not None: _validate_overlap_probe(probe,raw['stream_id'],tensor['transfer_bytes'],plan)
        resources=raw['resources']
        if resources['policy']!='advisory_only; occupancy_is_not_an_execution_gate': raise ValueError('invalid resource policy')
        for observation in (resources['before'],resources['after']):
            for key in ('cuda_allocated_bytes','cuda_reserved_bytes','cuda_free_bytes','cuda_total_bytes'):
                _integer(observation[key])
            if (observation['cuda_allocated_bytes']>observation['cuda_reserved_bytes']
                    or observation['cuda_free_bytes']>observation['cuda_total_bytes']): raise ValueError('invalid resource observation')
            cpu=observation['cpu']
            if cpu['status']=='measured':
                for key in ('rss_bytes','lifetime_hwm_bytes'): _integer(cpu['observation'][key])
                if cpu['observation']['rss_bytes']>cpu['observation']['lifetime_hwm_bytes']: raise ValueError('invalid RSS observation')
            elif cpu['status']!='unsupported' or cpu['observation'] is not None or not cpu['reason']:
                raise ValueError('invalid CPU resource availability')
        if raw['allocator_fraction']!=plan.cuda_memory_fraction or raw['cpu_memory_limit_bytes']!=plan.cpu_memory_limit_bytes:
            raise ValueError('resource quota mismatch')
        if raw['cleanup']!=dict(terminal_completed=True,process_owned_resources=True): raise ValueError('incomplete copy cleanup')
    if data['measurements']!=_transfer_metrics(rows,evidence_version=version,overlap_probe=probe): raise ValueError('copy metrics mismatch')


def _validate_overlap_probe(probe,copy_stream,transfer_bytes,plan):
    for k in ('copy_stream_id','compute_stream_id'): _integer(probe[k])
    if probe['copy_stream_id']!=copy_stream or probe['compute_stream_id']==copy_stream:
        raise ValueError('overlap probe requires distinct owned streams')
    if (probe['epoch_stream_id']!=copy_stream or probe['compute_waited_on_epoch'] is not True
            or probe['terminal_completed'] is not True or probe['round_trip_equal'] is not True
            or probe['compute_equal'] is not True or probe['events_preinitialized'] is not True):
        raise ValueError('incomplete overlap probe')
    if (probe['direction']!='H2D' or probe['transfer_bytes']!=transfer_bytes
            or probe['compute_shape']!=[256,256] or probe['compute_iterations']!=16
            or probe['host_pinned'] is not plan.pinned or probe['non_blocking_requested'] is not plan.non_blocking):
        raise ValueError('invalid overlap probe workload')
    for key in ('copy_start_ms','copy_end_ms','compute_start_ms','compute_end_ms'):
        v=probe[key]
        if type(v) not in (int,float) or not math.isfinite(v) or v<0: raise ValueError('invalid overlap probe time')
    if probe['copy_end_ms']<probe['copy_start_ms'] or probe['compute_end_ms']<probe['compute_start_ms']:
        raise ValueError('reversed overlap probe interval')
