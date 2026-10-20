"""P16 sampled native counters. No workload execution, CUDA calls or busy inference."""
from dataclasses import asdict,dataclass
import math
import os
import platform
import re
import time

from . import _telemetry
from .inference import _base
from .result import UtilizationResult,_integer,_positive


@dataclass(frozen=True)
class UtilizationProtocol:
    duration_seconds: float = 1.
    interval_seconds: float = .1
    max_samples: int = 128
    min_valid_intervals: int = 2
    gpu: bool = False
    gpu_selector: str = '0'
    sensor_timeout_seconds: float = 1.
    workload_label: str | None = None

    def __post_init__(self):
        for v in (self.duration_seconds,self.interval_seconds,self.sensor_timeout_seconds): _positive(v)
        if self.duration_seconds>60 or self.interval_seconds<.001 or self.interval_seconds>self.duration_seconds:
            raise ValueError('bounded duration <=60s and interval >=1ms <=duration')
        if self.sensor_timeout_seconds>5: raise ValueError('sensor timeout <=5s')
        _integer(self.max_samples,2); _integer(self.min_valid_intervals,1)
        if self.max_samples>10000 or self.min_valid_intervals>=self.max_samples: raise ValueError('sample limits')
        if type(self.gpu) is not bool: raise TypeError('gpu must be bool')
        if type(self.gpu_selector) is not str or len(self.gpu_selector)>96 or not re.fullmatch(
                r'(?:[0-9]+|GPU-[A-Za-z0-9-]+|[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f])',self.gpu_selector):
            raise ValueError('physical NVIDIA index/UUID/PCI selector required')
        if self.workload_label is not None and (type(self.workload_label) is not str or len(self.workload_label)>256):
            raise ValueError('bounded workload label required')


def _cpu_intervals(samples,cores,interval_ns):
    _integer(cores,1); rows=[]
    for a,b in zip(samples,samples[1:]):
        wall=b['timestamp_ns']-a['timestamp_ns']
        row=dict(start_ns=a['timestamp_ns'],end_ns=b['timestamp_ns'],wall_ns=wall,
            kernel_delta_ns=None,user_delta_ns=None,single_core_percent=None,machine_capacity_percent=None,
            system_busy_percent=None,system_status='unsupported',status='missing',reason='missing_counter',
            gap=wall>2*interval_ns)
        if a['status']=='measured' and b['status']=='measured':
            kernel=b['kernel_ns']-a['kernel_ns']; user=b['user_ns']-a['user_ns']
            if wall<=0: row.update(status='invalid',reason='nonpositive_wall_interval')
            elif min(kernel,user)<0: row.update(status='counter_reset',reason='negative_delta_no_wrap_inference')
            else:
                row.update(kernel_delta_ns=kernel,user_delta_ns=user,single_core_percent=100*(kernel+user)/wall,
                    machine_capacity_percent=100*(kernel+user)/wall/cores,status='measured',reason=None)
            if cores<=64 and a.get('system_status')=='measured' and b.get('system_status')=='measured':
                idle=b['system_idle_ns']-a['system_idle_ns']; k=b['system_kernel_ns']-a['system_kernel_ns']
                u=b['system_user_ns']-a['system_user_ns']; total=k+u
                if min(idle,k,u)>=0 and total>0 and idle<=total:
                    row.update(system_busy_percent=100*(total-idle)/total,system_status='measured')
        rows.append(row)
    return rows


def _utilization_metrics(cpu,gpu,cores,p):
    intervals=_cpu_intervals(cpu,cores,int(p.interval_seconds*1e9))
    valid=[r for r in intervals if r['status']=='measured']
    enough=len(valid)>=p.min_valid_intervals
    wall=sum(r['wall_ns'] for r in valid)
    used=sum(r['kernel_delta_ns']+r['user_delta_ns'] for r in valid)
    fields=('gpu_busy_percent','memory_controller_busy_percent','memory_used_bytes','memory_free_bytes','memory_total_bytes')
    sensors={}
    for field in fields:
        values=[r['values'][field] for r in gpu if r['status']=='measured' and r['values'][field] is not None]
        sensors[field]=dict(sample_count=len(values),sample_mean=math.fsum(values)/len(values) if values else None,
            status='measured' if len(values)>=p.min_valid_intervals else 'insufficient_samples' if values else 'unsupported',
            reason=None if len(values)>=p.min_valid_intervals else 'too_few_sensor_samples' if values else 'no_supported_sensor_values',
            unit='percent' if 'percent' in field else 'byte',aggregation='sample_arithmetic_mean_not_time_integral')
    unsupported=dict(value=None,status='unsupported',reason='not_provided_by_this_backend')
    return dict(status='measured' if enough and (not p.gpu or sensors['gpu_busy_percent']['status']=='measured') else 'partial',
        cpu=dict(status='measured' if enough else 'insufficient_samples',valid_intervals=len(valid),
            single_core_percent=100*used/wall if enough else None,
            machine_capacity_percent=100*used/wall/cores if enough else None,
            normalization='process(kernel+user)/wall; machine=single_core/logical_cores',logical_cores=cores,
            valid_wall_ns=wall,includes_sampler_overhead=True),
        gpu=dict(fields=sensors,domain='physical_device_wide',process_gpu_utilization=unsupported,
            tensor_core_utilization=dict(unsupported),sensor_window_seconds=None,
            sensor_window_status='not_exposed_by_CLI; product_dependent_1/6_to_1_second',
            attribution='not_exclusive_to_HYSo_or_workload',clock_domain='sensor_internal_window_unknown'),
        gap_count=sum(r['gap'] for r in intervals),missing_interval_count=sum(r['status']!='measured' for r in intervals))


def measure_utilization(*,protocol=UtilizationProtocol()):
    """Blocking bounded sampler of caller process CPU and optional physical GPU.

    Workload must run independently; label is caller-declared, not attribution.
    No writes occur in the window. GPU queries use installed nvidia-smi/NVML,
    one bounded subprocess per query; no Python NVML dependency is required.
    """
    if type(protocol) is not UtilizationProtocol: raise TypeError('UtilizationProtocol required')
    p=protocol; cpu=[]; gpu=[]; cores=os.cpu_count() or 1; start=time.perf_counter_ns()
    deadline=start+int(p.duration_seconds*1e9); due=start; query_ns=0
    while len(cpu)<p.max_samples:
        now=time.perf_counter_ns()
        if now<due: time.sleep(min((due-now)/1e9,max(0,(deadline-now)/1e9)))
        if time.perf_counter_ns()>=deadline: break
        a=time.perf_counter_ns()
        try: row=_telemetry.cpu_observe()
        except (OSError,NotImplementedError) as error:
            b=time.perf_counter_ns(); row=dict(timestamp_ns=(a+b)//2,query_start_ns=a,query_end_ns=b,
                kernel_ns=None,user_ns=None,status='unsupported',reason=type(error).__name__+': '+str(error))
        cpu.append(row)
        remaining=(deadline-time.perf_counter_ns())/1e9
        if p.gpu and remaining>0:
            gpu.append(_telemetry.gpu_observe(p.gpu_selector,min(p.sensor_timeout_seconds,remaining)))
        query_ns+=time.perf_counter_ns()-a
        due=max(due+int(p.interval_seconds*1e9),time.perf_counter_ns())
    end=time.perf_counter_ns()
    data=_base('utilization','P16',dict(id='native_sampled_counters',version=1,
        scope='current_process_CPU_optional_physical_GPU',configuration=asdict(p)),
        dict(model={'kind':'not_applicable'},input={'kind':'not_applicable'},environment=dict(device='cpu',
            platform=platform.platform(),python=platform.python_version(),pid=os.getpid(),
            package_source=__file__,cpu_backend='windows.GetProcessTimes/GetSystemTimes',
            gpu_backend='installed_nvidia-smi.NVML' if p.gpu else 'not_requested')))
    data.update(requested=dict(duration_seconds=p.duration_seconds,interval_seconds=p.interval_seconds,max_samples=p.max_samples),
        actual=dict(cpu_samples=len(cpu),gpu_queries=len(gpu),start_ns=start,end_ns=end,elapsed_seconds=(end-start)/1e9,
            stop_reason='sample_budget' if len(cpu)==p.max_samples else 'deadline'),
        measurements=_utilization_metrics(cpu,gpu,cores,p),raw_evidence=dict(cpu_samples=cpu,gpu_samples=gpu,
            cpu_intervals=_cpu_intervals(cpu,cores,int(p.interval_seconds*1e9)),logical_cores=cores,
            clock='perf_counter_ns_current_process; GPU tool_wall_clock_separate',
            query_overhead_ns=query_ns,overhead_scope='CPU+sensor_queries_including_subprocess_wait_not_sleep',
            workload=dict(label=p.workload_label,overlap='caller_declared_unverified',exclusive=False)))
    data['diagnostics']['limitations']=['Samples are not interpolation or continuous/time-weighted busy integral.',
        'Device GPU busy is not Tensor Core usage; memory occupancy is not compute utilization.',
        'Sensor internal windows may exceed short workloads and span outside collection; no exact workload attribution.',
        'System CPU scope >64 logical cores unsupported; process normalization uses OS logical count.',
        'OS CPU adapter Windows-only; GPU selector uses NVIDIA physical enumeration, not CUDA_VISIBLE_DEVICES.']
    return UtilizationResult(data)


def _validate_utilization(data):
    p=UtilizationProtocol(**data['protocol']['configuration']); raw=data['raw_evidence']; actual=data['actual']
    cpu=raw['cpu_samples']; gpu=raw['gpu_samples']; cores=raw['logical_cores']; _integer(cores,1)
    if len(cpu)>p.max_samples or len(gpu)>len(cpu) or (not p.gpu and gpu): raise ValueError('sample budget mismatch')
    for row in cpu+gpu:
        for key in ('query_start_ns','query_end_ns'): _integer(row[key])
        if row['query_end_ns']<row['query_start_ns']: raise ValueError('query clock reversal')
    for row in cpu:
        if not row['query_start_ns']<=row['timestamp_ns']<=row['query_end_ns']: raise ValueError('CPU midpoint clock')
        if row['status']=='measured':
            _integer(row['kernel_ns']); _integer(row['user_ns'])
    identities={r['values']['uuid'] for r in gpu if r['status']=='measured'}
    if len(identities)>1: raise ValueError('physical GPU identity changed within window')
    for row in gpu:
        if row['status']=='measured':
            for field,v in row['values'].items():
                if field.endswith(('_percent','_bytes')) and v is not None:
                    if type(v) not in (float,int) or not math.isfinite(v) or v<0 or (field.endswith('_percent') and v>100):
                        raise ValueError('invalid sensor numeric value')
    if actual['cpu_samples']!=len(cpu) or actual['gpu_queries']!=len(gpu): raise ValueError('sample count mismatch')
    _integer(actual['start_ns']); _integer(actual['end_ns'])
    if actual['end_ns']<actual['start_ns'] or actual['elapsed_seconds']!=(actual['end_ns']-actual['start_ns'])/1e9:
        raise ValueError('collection clock mismatch')
    if any(r['query_start_ns']<actual['start_ns'] or r['query_end_ns']>actual['end_ns'] for r in cpu+gpu):
        raise ValueError('samples outside window')
    if raw['cpu_intervals']!=_cpu_intervals(cpu,cores,int(p.interval_seconds*1e9)): raise ValueError('CPU interval mismatch')
    if data['measurements']!=_utilization_metrics(cpu,gpu,cores,p): raise ValueError('utilization metrics mismatch')
    if data['requested']!=dict(duration_seconds=p.duration_seconds,interval_seconds=p.interval_seconds,max_samples=p.max_samples):
        raise ValueError('requested sampler mismatch')
