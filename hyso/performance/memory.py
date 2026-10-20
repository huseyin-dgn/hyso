"""Native process RSS observations and independent logical/artifact footprint.

No inference, CUDA runtime calls, allocator resets or global GC manipulation.
Memory sampling is current-process instrumentation, not an isolated model peak.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import platform
import random
import re
import tempfile
import threading
import time

import numpy as np
import torch
from torch import nn

from hyso.benchmark.runner import _counts, _descriptor, _bits, _RUN_LOCK
from .inference import _base
from .result import MemoryResult, FootprintResult, _integer, _positive


class _BackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class MemoryProtocol:
    interval_seconds: float = .01
    max_samples: int = 10000

    def __post_init__(self):
        _positive(self.interval_seconds)
        if not math.isfinite(self.interval_seconds * 1e9) or self.interval_seconds < 1e-9:
            raise ValueError('interval must be representable in nanoseconds')
        _integer(self.max_samples, 2)


def _linux_status(text):
    values = {}
    for line in text.splitlines():
        key = line.split(':', 1)[0]
        if key in ('VmRSS', 'VmHWM'):
            match = re.fullmatch(r'(VmRSS|VmHWM):\s+(\d+)\s+kB\s*', line)
            if match is None or key in values:
                raise _BackendUnavailable('invalid /proc/self/status RSS/HWM fields or units')
            values[key] = int(match[2]) * 1024
    if set(values) != {'VmRSS', 'VmHWM'} or values['VmHWM'] < values['VmRSS']:
        raise _BackendUnavailable('missing or inconsistent /proc RSS/HWM')
    return values['VmRSS'], values['VmHWM']


@lru_cache(maxsize=1)
def _windows_backend():
    # ctypes caches POINTER(Structure) globally: defining a new Structure per
    # sample would permanently grow that cache and pollute the measured RSS.
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    api = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.argtypes = []
    api.GetProcessMemoryInfo.restype = wintypes.BOOL
    api.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    return kernel.GetCurrentProcess, api.GetProcessMemoryInfo, Counters


def _windows_rss():
    import ctypes
    get_process, read, Counters = _windows_backend()
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not read(get_process(), ctypes.byref(counters), counters.cb):
        raise _BackendUnavailable('GetProcessMemoryInfo failed: ' + str(ctypes.get_last_error()))
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def _observe():
    start = time.perf_counter_ns()
    system = platform.system()
    try:
        if system == 'Windows':
            rss, hwm = _windows_rss()
            backend = 'windows.GetProcessMemoryInfo.working_set'
        elif system == 'Linux':
            rss, hwm = _linux_status(Path('/proc/self/status').read_text(encoding='ascii'))
            backend = 'linux.proc_self_status'
        else:
            raise _BackendUnavailable('unsupported OS: ' + system)
    except OSError as error:
        raise _BackendUnavailable(system + ' RSS backend unavailable: ' + str(error)) from error
    end = time.perf_counter_ns()
    _integer(rss); _integer(hwm)
    if hwm < rss or end < start:
        raise ValueError('inconsistent native memory observation/clock')
    return dict(timestamp_ns=(start + end)//2, query_start_ns=start, query_end_ns=end,
                rss_bytes=rss, lifetime_hwm_bytes=hwm, backend=backend)


def _metric(value, unit='byte', status='measured', reason=None):
    return dict(value=value, unit=unit, status=status, reason=reason)


def _environment():
    return dict(device='cpu', platform=platform.system(), platform_detail=platform.platform(),
                python=platform.python_version(), torch=str(torch.__version__), pid=os.getpid(),
                process_scope='current_process', source_revision=None, revision_status='not_collected')


def _memory_metrics(observations, reason=None):
    names = ('baseline_rss_bytes', 'current_rss_bytes', 'sampled_peak_rss_bytes',
             'baseline_lifetime_hwm_bytes', 'current_lifetime_hwm_bytes')
    if not observations:
        return {name: _metric(None, status='unsupported', reason=reason) for name in names}
    values = [observations[0]['rss_bytes'], observations[-1]['rss_bytes'],
              max(row['rss_bytes'] for row in observations), observations[0]['lifetime_hwm_bytes'],
              observations[-1]['lifetime_hwm_bytes']]
    return dict(zip(names, map(_metric, values)))


def measure_memory(workload=None, *, protocol=MemoryProtocol()):
    """Observe current process RSS; optionally sample while a synchronous callable runs.

    None performs one native snapshot. Callable return values are discarded before
    the final observation. A sampler thread is always stopped/joined on failure.
    The workload owns its mutations/resources; no model transaction is promised.
    Python/NumPy/Torch CPU RNG are restored, under caller-exclusive process usage.
    Unsupported backend returns null/status/reason without executing the workload.
    """
    if type(protocol) is not MemoryProtocol:
        raise TypeError('protocol must be MemoryProtocol')
    if workload is not None and not callable(workload):
        raise TypeError('workload must be a synchronous callable or None')
    import inspect
    if workload is not None and (inspect.iscoroutinefunction(workload) or
            inspect.iscoroutinefunction(getattr(workload, '__call__', None))):
        raise TypeError('async workloads are unsupported')
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError('concurrent/reentrant benchmark/performance calls unsupported')
    rng = None
    stop = threading.Event()
    thread = None
    observations = []
    errors = []
    reason = None
    try:
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        start = time.perf_counter_ns()
        try:
            observations.append(_observe())
        except _BackendUnavailable as error:
            reason = str(error)
        if reason is None and workload is not None:
            def sample():
                while not stop.wait(protocol.interval_seconds):
                    if len(observations) >= protocol.max_samples - 1:
                        errors.append(RuntimeError('memory sample budget exhausted'))
                        return
                    try:
                        observations.append(_observe())
                    except BaseException as error:
                        errors.append(error)
                        return
            thread = threading.Thread(target=sample, name='hyso-rss-sampler', daemon=True)
            thread.start()
            try:
                returned = workload()
                if inspect.isawaitable(returned) or inspect.isgenerator(returned):
                    if hasattr(returned, 'close'): returned.close()
                    raise TypeError('workload must complete synchronously, not return an iterator/awaitable')
                del returned
            finally:
                stop.set()
                thread.join()
            if errors:
                raise RuntimeError('native RSS sampler failed') from errors[0]
            observations.append(_observe())
        end = time.perf_counter_ns()
        gaps = [b['timestamp_ns'] - a['timestamp_ns'] for a,b in zip(observations, observations[1:])]
        data = _base('memory', 'P10', dict(id='process_rss', version=1, scope='current_process_rss',
                     clock='perf_counter_ns', configuration=asdict(protocol),
                     workload='snapshot' if workload is None else 'synchronous_callable',
                     included=['native_OS_queries', 'sampler_thread_and_GIL_overhead', 'workload'],
                     excluded=['export', 'artifact_hashing', 'tracking'], lifetime_peak_reset=False),
                     dict(environment=_environment(), model=dict(status='not_applicable', reason='process_scope'),
                          input=dict(source='none' if workload is None else 'caller_callable')))
        data.update(requested=dict(interval_seconds=protocol.interval_seconds, max_samples=protocol.max_samples),
                    actual=dict(observations=len(observations), start_ns=start, end_ns=end,
                                duration_seconds=(end-start)/1e9),
                    measurements=_memory_metrics(observations, reason),
                    raw_evidence=dict(observations=observations, gaps_ns=gaps))
        data['diagnostics']['limitations'] = ['Sampled peak is not an absolute window peak; short transients may be missed.',
            'OS lifetime high-water mark is never reset and includes pre-window allocations.',
            'RSS is process-wide including Torch/native/Python/runtime/sampler pages, not model-only RAM.',
            'Sampling has native-query, Python/GIL and scheduling overhead; gaps are actual timestamps.',
            'No worker isolation/timeout is implied; callable owns cleanup and must terminate.',
            'Linux adapter is protocol-tested; native validation must be recorded per OS.']
        return MemoryResult(data)
    finally:
        stop.set()
        try:
            if thread is not None and thread.ident is not None:
                thread.join()
        finally:
            try:
                if rng is not None:
                    random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
            finally:
                _RUN_LOCK.release()


def _profile_footprint(model):
    if not isinstance(model, nn.Module):
        raise TypeError('model must be nn.Module')
    modules, entries, seen = [], [], set()
    stack = [('', model, frozenset())]
    while stack:
        path, module, ancestors = stack.pop()
        if id(module) in ancestors:
            raise ValueError('cyclic module graph unsupported')
        if id(module) in seen:
            continue
        seen.add(id(module)); modules.append((path,module))
        if (type(module).state_dict is not nn.Module.state_dict or 'state_dict' in vars(module)
                or type(module).get_extra_state is not nn.Module.get_extra_state
                or (type(module)._save_to_state_dict is not nn.Module._save_to_state_dict and
                    not type(module).__module__.startswith('torch.nn.'))
                or any(bool(value) for name,value in vars(module).items() if name.endswith('_hooks') and isinstance(value,dict))):
            raise ValueError('custom state_dict/extra state/hooks unsupported')
        for kind, mapping in (('parameter', module._parameters), ('buffer', module._buffers)):
            for name,tensor in mapping.items():
                if tensor is None: continue
                if (type(tensor) not in (torch.Tensor, nn.Parameter) or tensor.device.type != 'cpu'
                        or tensor.layout != torch.strided or tensor.is_quantized or tensor.is_conj() or tensor.is_neg()):
                    raise ValueError('footprint requires initialized ordinary dense CPU tensors')
                entries.append((kind, path+'.'+name if path else name, tensor))
        for name, child in reversed(list(module._modules.items())):
            if child is not None:
                stack.append((path+'.'+name if path else name, child, ancestors | {id(module)}))
    return modules, entries


def _file_info(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('artifact must be a regular, non-symlink file')
    digest = hashlib.sha256()
    count = 0
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        while True:
            chunk = handle.read(1024*1024)
            if not chunk: break
            count += len(chunk); digest.update(chunk)
        after = os.fstat(handle.fileno())
    published = path.stat()
    signature = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    if path.is_symlink() or signature(before) != signature(after) or signature(after) != signature(published) or count != after.st_size:
        raise ValueError('artifact changed during streamed size/SHA verification')
    if count == 0:
        raise ValueError('empty artifact unsupported')
    return dict(path=str(path.resolve()), size_bytes=after.st_size, sha256=digest.hexdigest())


def _serialize(model, destination, verify, overwrite):
    destination = Path(destination)
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ValueError('destination must be a regular non-symlink file')
    if destination.exists() and not overwrite:
        raise FileExistsError(str(destination))
    temporary = None
    try:
        state = model.state_dict()
        if any(type(tensor) is not torch.Tensor for tensor in state.values()):
            raise ValueError('only tensor state_dict values are supported')
        with tempfile.NamedTemporaryFile(mode='w+b', prefix='.hyso-footprint-', suffix='.tmp',
                                         dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(state, handle)
            handle.flush(); os.fsync(handle.fileno())
        if verify:
            # Only our own just-written file, never arbitrary caller checkpoints.
            loaded = torch.load(temporary, map_location='cpu', weights_only=True)
            if (type(loaded) is not type(state) or list(loaded) != list(state) or any(
                    type(loaded[key]) is not torch.Tensor or loaded[key].dtype != tensor.dtype
                    or loaded[key].shape != tensor.shape or loaded[key].stride() != tensor.stride()
                    or not torch.equal(_bits(loaded[key]), _bits(tensor)) for key,tensor in state.items())):
                raise ValueError('state_dict round-trip mismatch')
        info = _file_info(temporary)
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
        final = _file_info(destination)
        if (info['size_bytes'],info['sha256']) != (final['size_bytes'],final['sha256']):
            raise ValueError('published artifact differs from serialized bytes')
        return final
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def measure_footprint(model, *, artifact_path=None, serialize_to=None, artifact_format=None,
                      verify_roundtrip=False, expected_sha256=None, overwrite=False):
    """Logical dense CPU model bytes plus an optional real artifact.

    Existing artifacts are opaque byte observations: format is caller-declared,
    never deserialized or asserted to match the model. Only explicit serialize_to
    creates a PyTorch state_dict; optional round-trip reads that trusted temp file
    with weights_only=True. Future formats can associate bytes without conversion.
    """
    if type(verify_roundtrip) is not bool or type(overwrite) is not bool:
        raise TypeError('verify_roundtrip/overwrite must be bool')
    if artifact_path is not None and serialize_to is not None:
        raise ValueError('choose existing artifact_path or explicit serialize_to')
    if verify_roundtrip and serialize_to is None:
        raise ValueError('round-trip verification is only for our own explicit serialization')
    if artifact_format is not None and (type(artifact_format) is not str or not artifact_format.strip()):
        raise ValueError('artifact_format must be a nonempty explicit format identifier')
    if artifact_path is not None and artifact_format is None:
        raise ValueError('existing artifact requires caller-declared artifact_format')
    if serialize_to is not None and artifact_format not in (None,'pytorch.state_dict'):
        raise ValueError('explicit serialization supports only pytorch.state_dict')
    if expected_sha256 is not None and (artifact_path is None or type(expected_sha256) is not str or
                                       re.fullmatch('[0-9a-fA-F]{64}',expected_sha256) is None):
        raise ValueError('expected_sha256 requires existing artifact and 64 hex characters')
    if overwrite and serialize_to is None:
        raise ValueError('overwrite requires serialize_to')
    if artifact_format is not None and artifact_path is None and serialize_to is None:
        raise ValueError('artifact format requires an artifact')
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError('concurrent/reentrant benchmark/performance calls unsupported')
    rng = None
    try:
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        modules, entries = _profile_footprint(model)
        descriptor, weights_hash = _descriptor(modules, entries)
        logical = _counts(entries)
        logical.update(unique_parameter_tensors=len({id(t) for k,_,t in entries if k=='parameter'}),
                       unique_buffer_tensors=len({id(t) for k,_,t in entries if k=='buffer'}),
                       buffer_elements=sum(t.numel() for t in {id(t):t for k,_,t in entries if k=='buffer'}.values()))
        for row,(_,_,tensor) in zip(descriptor['tensors'], entries): row['element_size_bytes']=tensor.element_size()
        artifact = dict(status='disabled', reason='no_artifact_requested', path=None, size_bytes=None, sha256=None,
                        format=None, format_validation='not_applicable', model_match='not_verified', ownership='caller')
        if serialize_to is not None:
            info = _serialize(model, serialize_to, verify_roundtrip, overwrite)
            artifact.update(info, status='measured',reason=None,format='pytorch.state_dict',
                            format_validation='roundtrip_verified' if verify_roundtrip else 'produced_not_roundtripped',
                            model_match='roundtrip_verified' if verify_roundtrip else 'produced_from_model')
        elif artifact_path is not None:
            info = _file_info(artifact_path)
            if expected_sha256 is not None and info['sha256'] != expected_sha256.lower():
                raise ValueError('artifact SHA256 mismatch')
            artifact.update(info,status='measured',reason=None,format=artifact_format,
                            format_validation='caller_declared_not_deserialized')
        metrics = _footprint_metrics(logical, artifact)
        data = _base('footprint','P11',dict(id='logical_and_artifact_bytes',version=1,scope='dense_cpu_model_and_optional_artifact',
            byte_accounting='unique_tensor_object_logical', serialization='explicit_state_dict' if serialize_to is not None else 'none',
            verify_roundtrip=verify_roundtrip),dict(model=dict(qualified_class=type(model).__module__+'.'+type(model).__qualname__,
            structural_descriptor=descriptor,weights_sha256=weights_hash),input=dict(source='registered_cpu_tensors'),environment=_environment()))
        data.update(requested=dict(artifact=artifact_path is not None or serialize_to is not None,verify_roundtrip=verify_roundtrip),
                    actual=dict(tensor_entries=len(entries),artifacts=int(artifact['status']=='measured')),
                    measurements=metrics,raw_evidence=dict(logical=logical,artifact=artifact))
        data['diagnostics']['limitations']=['Logical object bytes are neither physical storage nor process RSS.',
            'Parameter/buffer objects are deduplicated per category; cross-category union is deduplicated separately.',
            'Distinct view objects sharing storage are counted separately as in Benchmark schema1.',
            'Nonpersistent buffers count logically but are excluded from serialized state_dict.',
            'Existing artifacts have measured size/hash, caller-declared format, and unverified model association.',
            'No CUDA model movement/synchronization, conversion engine or unsafe checkpoint deserialization.']
        return FootprintResult(data)
    finally:
        try:
            if rng is not None:
                random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        finally:
            _RUN_LOCK.release()


def _footprint_metrics(logical, artifact):
    names=('total_parameters','trainable_parameters','frozen_parameters','unique_parameter_tensors',
           'unique_buffer_tensors','buffer_elements','parameter_bytes','buffer_bytes','parameter_buffer_bytes')
    metrics={name:_metric(logical[name], unit='byte' if name.endswith('bytes') else 'count',status='derived') for name in names}
    metrics['serialized_artifact_bytes']=_metric(artifact['size_bytes'],status=artifact['status'],reason=artifact['reason'])
    metrics['physical_memory_bytes']=_metric(None,status='unsupported',reason='physical_memory_not_measured')
    return metrics


def _validate_memory(data):
    plan=MemoryProtocol(**data['protocol']['configuration'])
    protocol=data['protocol']
    if (protocol['id']!='process_rss' or protocol['scope']!='current_process_rss' or
            protocol['clock']!='perf_counter_ns' or protocol['lifetime_peak_reset'] is not False):
        raise ValueError('invalid process RSS protocol')
    _integer(data['provenance']['environment']['pid'],1)
    actual=data['actual']
    for name in ('observations','start_ns','end_ns'): _integer(actual[name])
    if actual['end_ns']<actual['start_ns'] or actual['duration_seconds']!=(actual['end_ns']-actual['start_ns'])/1e9:
        raise ValueError('invalid memory window duration')
    rows=data['raw_evidence']['observations']
    if actual['observations']!=len(rows) or len(rows)>plan.max_samples:
        raise ValueError('inconsistent observation count/budget')
    last=actual['start_ns']
    backend=None
    hwm=0
    for row in rows:
        for name in ('timestamp_ns','query_start_ns','query_end_ns','rss_bytes','lifetime_hwm_bytes'): _integer(row[name])
        if (not last<=row['query_start_ns']<=row['timestamp_ns']<=row['query_end_ns']<=actual['end_ns']
                or row['lifetime_hwm_bytes']<max(hwm,row['rss_bytes'])):
            raise ValueError('invalid sample timestamps or lifetime high-water mark')
        if backend is not None and row['backend']!=backend: raise ValueError('backend changed within window')
        if row['backend'] not in ('windows.GetProcessMemoryInfo.working_set','linux.proc_self_status'):
            raise ValueError('unknown native RSS backend')
        backend=row['backend']; last=row['query_end_ns']; hwm=row['lifetime_hwm_bytes']
    gaps=[b['timestamp_ns']-a['timestamp_ns'] for a,b in zip(rows,rows[1:])]
    if data['raw_evidence']['gaps_ns']!=gaps: raise ValueError('inconsistent sampling gaps')
    if data['requested']!=dict(interval_seconds=plan.interval_seconds,max_samples=plan.max_samples):
        raise ValueError('inconsistent requested memory protocol')
    reason=None if rows else data['measurements']['current_rss_bytes']['reason']
    if not rows and (type(reason) is not str or not reason): raise ValueError('unsupported memory needs explicit reason')
    if data['measurements']!=_memory_metrics(rows,reason): raise ValueError('RSS values disagree with evidence')


def _validate_footprint(data):
    if data['protocol']['id']!='logical_and_artifact_bytes' or data['protocol']['scope']!='dense_cpu_model_and_optional_artifact':
        raise ValueError('invalid footprint protocol')
    if (type(data['protocol']['verify_roundtrip']) is not bool or
            data['protocol']['serialization'] not in ('none','explicit_state_dict') or
            data['provenance']['environment']['device']!='cpu'):
        raise ValueError('invalid footprint runtime/serialization policy')
    for value in data['actual'].values(): _integer(value)
    logical=data['raw_evidence']['logical']
    artifact=data['raw_evidence']['artifact']
    rows=data['provenance']['model']['structural_descriptor']['tensors']
    parameters={}; buffers={}
    for row in rows:
        for size in row['shape']: _integer(size)
        _integer(row['element_size_bytes'],1); _integer(row['object_alias']); _integer(row['storage_alias'])
        bucket=parameters if row['kind']=='parameter' else buffers if row['kind']=='buffer' else None
        if bucket is None: raise ValueError('invalid tensor kind')
        value=(math.prod(row['shape']),row['element_size_bytes'],row['requires_grad'],row['storage_alias'])
        if row['object_alias'] in bucket and bucket[row['object_alias']]!=value: raise ValueError('inconsistent object alias')
        bucket[row['object_alias']]=value
    combined=dict(parameters); combined.update(buffers)
    expected=dict(total_parameters=sum(v[0] for v in parameters.values()),
        trainable_parameters=sum(v[0] for v in parameters.values() if v[2]),
        parameter_bytes=sum(v[0]*v[1] for v in parameters.values()),buffer_bytes=sum(v[0]*v[1] for v in buffers.values()),
        parameter_buffer_bytes=sum(v[0]*v[1] for v in combined.values()),
        unique_parameter_tensors=len(parameters),unique_buffer_tensors=len(buffers),buffer_elements=sum(v[0] for v in buffers.values()),
        byte_accounting='unique_tensor_object_logical',cross_category_object_alias=bool(parameters.keys() & buffers.keys()),
        distinct_objects_share_storage=len({v[3] for v in combined.values()})<len(combined))
    expected['frozen_parameters']=expected['total_parameters']-expected['trainable_parameters']
    if logical!=expected: raise ValueError('logical footprint disagrees with descriptor')
    for name,value in logical.items():
        if name.endswith(('bytes','parameters','tensors','elements')): _integer(value)
    if artifact['status']=='measured':
        _integer(artifact['size_bytes'],1)
        if (not artifact['path'] or type(artifact['sha256']) is not str or re.fullmatch('[0-9a-f]{64}',artifact['sha256']) is None
                or type(artifact['format']) is not str or not artifact['format'].strip() or artifact['reason'] is not None):
            raise ValueError('invalid artifact identity/size/hash/format')
        if data['protocol']['serialization']=='explicit_state_dict':
            verified=data['protocol']['verify_roundtrip']
            if (artifact['format']!='pytorch.state_dict' or
                    artifact['format_validation']!=('roundtrip_verified' if verified else 'produced_not_roundtripped') or
                    artifact['model_match']!=('roundtrip_verified' if verified else 'produced_from_model')):
                raise ValueError('inconsistent owned serialization verification')
        elif (data['protocol']['verify_roundtrip'] or artifact['model_match']!='not_verified' or
              artifact['format_validation']!='caller_declared_not_deserialized'):
            raise ValueError('caller artifact cannot claim verified model/format')
    elif (artifact['status']!='disabled' or artifact['reason']!='no_artifact_requested' or
          any(artifact[key] is not None for key in ('path','size_bytes','sha256','format'))):
        raise ValueError('missing artifact cannot have invented size/hash')
    if artifact['ownership']!='caller': raise ValueError('footprint artifact must remain caller-owned')
    if data['measurements']!=_footprint_metrics(logical,artifact): raise ValueError('footprint metrics disagree with evidence')
    if data['actual']!=dict(tensor_entries=len(rows),artifacts=int(artifact['status']=='measured')):
        raise ValueError('inconsistent footprint observation count')
    if data['requested']!=dict(artifact=artifact['status']=='measured',verify_roundtrip=data['protocol']['verify_roundtrip']):
        raise ValueError('inconsistent footprint request')
