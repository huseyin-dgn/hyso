"""P18 same-capture analytic work / measured intervals; no hardware-counter fiction.

Tensor payload estimates are neither DRAM traffic nor lower bounds on it.
Optional caller-supplied theoretical roofs are scenario references, not verified
calibrations. P15 transfer rates and P10/P23 allocator/RSS bytes are not traffic.
"""
from dataclasses import dataclass,asdict
import hashlib
import json
import math

from .cold import ColdStartRecipe
from .profiling import ProfilingProtocol,profile_operators,analyze_flops,profile_layers
from .inference import _base
from .result import OperatorProfileResult,RooflineResult,_positive,_integer


def _hash(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RooflineProtocol:
    traffic: str = 'dispatch_payload_estimate'

    def __post_init__(self):
        if self.traffic not in ('dispatch_payload_estimate','unsupported'):
            raise ValueError('traffic must be dispatch_payload_estimate or unsupported')


@dataclass(frozen=True)
class RooflineReference:
    """Unverified theoretical scenario, tied to exactly one captured context.

    No calibration or hardware-limit validity is asserted from caller numbers.
    Consequently this reference cannot license a bottleneck or headroom claim.
    """
    profile_sha256: str
    compute_flops_per_second: float
    memory_bytes_per_second: float
    source: str

    def __post_init__(self):
        if type(self.profile_sha256) is not str or len(self.profile_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.profile_sha256):
            raise ValueError('source profile SHA256 required')
        _positive(self.compute_flops_per_second);_positive(self.memory_bytes_per_second)
        if type(self.source) is not str or not self.source.strip() or len(self.source)>4096:
            raise ValueError('bounded theoretical reference source required')

    @classmethod
    def for_profile(cls,profile,*,compute_flops_per_second,memory_bytes_per_second,source):
        return cls(_hash(_profile(profile)),compute_flops_per_second,memory_bytes_per_second,source)


def _math(flops,seconds,byte_count=None,compute=None,bandwidth=None):
    """Pure arithmetic, with SI bytes/FLOPs/seconds; no evidence classification."""
    if type(flops) not in (int,float) or not math.isfinite(flops) or flops<0:
        raise ValueError('finite nonnegative logical FLOPs required')
    _positive(seconds)
    if byte_count is not None:_positive(byte_count)
    if (compute is None)!=(bandwidth is None):raise ValueError('both scenario roofs required')
    if compute is not None:_positive(compute);_positive(bandwidth)
    rate=flops/seconds;intensity=flops/byte_count if byte_count is not None else None
    roof=min(compute,bandwidth*intensity) if compute is not None and intensity is not None else None
    values=dict(achieved_flops_per_second=rate,achieved_gflops_per_second=rate/1e9,
        arithmetic_intensity_flops_per_byte=intensity,scenario_roof_flops_per_second=roof,
        scenario_ridge_flops_per_byte=compute/bandwidth if compute is not None else None)
    if any(v is not None and not math.isfinite(v) for v in values.values()):raise ValueError('roofline arithmetic overflow')
    return values


_SIZES={'torch.bool':1,'torch.uint8':1,'torch.int8':1,'torch.int16':2,'torch.int32':4,'torch.int64':8,
        'torch.float16':2,'torch.bfloat16':2,'torch.float32':4,'torch.float64':8,'torch.complex64':8,'torch.complex128':16}


def _payload(value):
    if type(value) is dict:
        if value.get('kind')=='unrecorded':
            return 0 if value.get('type') in ('device','dtype','layout','memory_format') else None
        if value.get('kind')=='tensor':
            size=_SIZES.get(value.get('dtype'));shape=value.get('shape')
            if size is None or type(shape) is not list:return None
            for dimension in shape:_integer(dimension)
            return math.prod(shape)*size
        values=[_payload(v) for v in value.values()]
    elif type(value) is list:values=[_payload(v) for v in value]
    else:return 0
    return None if any(v is None for v in values) else sum(values)


def _unsupported(reason):return dict(value=None,status='unsupported',reason=reason)


def _profile(source):
    if type(source) is not OperatorProfileResult:
        raise TypeError('same-capture OperatorProfileResult required; independent timing/transfer/memory results cannot substitute')
    return OperatorProfileResult(source.to_dict()).to_dict()


def _derive(source,protocol,reference):
    profile=OperatorProfileResult(source)
    count=analyze_flops(profile).to_dict()['measurements']
    forwards=source['actual']['measured_forwards'];device=source['provenance']['environment']['device']
    if device=='cpu':
        times=[v/1000 for v in source['measurements']['overhead']['instrumented_host_forward_windows_ms']]
        clock='worker_perf_counter_ns';scope='instrumented_synchronous_host_forward_including_observer_and_hooks'
    else:
        layers=profile_layers(profile).to_dict()
        roots=[r for r in layers['raw_evidence']['invocations'] if r['parent_invocation_id'] is None]
        valid=(len(roots)==forwards and [r['measurement_index'] for r in roots]==list(range(forwards)) and
               all(r.get('cuda_elapsed_status')=='measured' for r in roots))
        times=[r['cuda_elapsed_us']/1e6 for r in roots] if valid else []
        clock='torch.cuda.Event.elapsed_time_ms';scope='instrumented_root_module_selected_stream_elapsed; not_kernel_only'
    timed=len(times)==forwards and all(type(v) in (float,int) and math.isfinite(v) and v>0 for v in times)
    seconds=math.fsum(times) if timed else None
    total=sum(r['total_flops'] for r in count['per_forward']) if count['status']=='complete' else None
    supported=count['supported_subtotal_flops']
    byte_counts=[0]*forwards;byte_valid=protocol.traffic=='dispatch_payload_estimate'
    if byte_valid:
        for call in source['raw_evidence']['aten_calls']:
            read,write=_payload(call['arguments']),_payload(call['output'])
            if read is None or write is None:byte_valid=False;break
            byte_counts[call['measurement_index']]+=read+write
    byte_total=sum(byte_counts) if byte_valid else None
    arithmetic=_math(total,seconds,byte_total if byte_total else None) if total is not None and timed else None
    intensity=_math(total,1.,byte_total)['arithmetic_intensity_flops_per_byte'] if total is not None and byte_total else None
    reference_data=_unsupported('no_verified_hardware_compute_or_memory_roof')
    if reference is not None:
        if reference.profile_sha256!=_hash(source):raise ValueError('roof reference belongs to another profile/context')
        reference_data=dict(status='estimated',kind='caller_supplied_theoretical_scenario',verification='unverified',
            **asdict(reference),scenario_roof_flops_per_second=_math(total,1.,byte_total if byte_total else None,
                reference.compute_flops_per_second,reference.memory_bytes_per_second)['scenario_roof_flops_per_second']
                if total is not None else None,
            ridge_flops_per_byte=reference.compute_flops_per_second/reference.memory_bytes_per_second)
    rate=(dict(status='derived',value=arithmetic['achieved_gflops_per_second'],unit='GFLOP/s',
               reason=None,scope='analytic_logical_FLOPs_divided_by_same_capture_instrumented_elapsed') if arithmetic is not None else
          dict(status='partial' if timed else 'unsupported',value=None,unit='GFLOP/s',
               reason='unmodeled_observed_operations' if timed else 'no_verified_same_capture_elapsed',scope=scope))
    rate['supported_subtotal_gflops_per_second']=_math(supported,seconds)['achieved_gflops_per_second'] if timed else None
    return dict(status='partial',core_status='derived' if arithmetic is not None else 'partial',
        logical_flops=dict(status='estimated' if total is not None else 'partial',value=count['total_flops_per_batch'],
            unit='FLOP/batch',supported_subtotal_per_batch=count['supported_flops_per_batch'],coverage=count['coverage'],
            convention='P12 analytic logical operations; FMA=2; not hardware instructions',per_forward=count['per_forward']),
        timing=dict(status='measured' if timed else 'unsupported',clock=clock,scope=scope,
            per_forward_seconds=times,total_seconds=seconds,forwards=forwards),
        achieved_compute=rate,
        traffic=dict(status='estimated' if byte_valid else 'unsupported',value=byte_total/forwards if byte_valid else None,
            unit='byte/batch',method=protocol.traffic,per_forward_bytes=byte_counts if byte_valid else None,
            domain='hypothetical_tensor_payload_reads_plus_writes_per_observed_dispatch; not_DRAM',
            reason=None if byte_valid else 'not_requested_or_tensor_dtype_metadata_unavailable'),
        arithmetic_intensity=dict(value=intensity,
            status='estimated' if intensity is not None else 'unsupported',unit='FLOP/byte',
            reason=None if intensity is not None else 'complete_FLOP_count_and_positive_byte_estimate_required',
            scope='logical_dispatch_payload_model_not_measured_DRAM_traffic'),
        position=dict(x_flops_per_byte=intensity,
            y_gflops_per_second=rate['value'],x_status='estimated' if intensity is not None else 'unsupported',
            y_status=rate['status']),
        theoretical_reference=reference_data,
        hardware_roofs=dict(compute_flops_per_second=_unsupported('no_verified_hardware_compute_roof'),
            memory_bytes_per_second=_unsupported('no_verified_device_DRAM_roof; P15_PCIe_transfers_excluded')),
        hardware_roof_calibration=_unsupported('no_validated_compute_and_device_DRAM_calibration_evidence'),
        bottleneck=_unsupported('estimated_payload_is_not_DRAM_traffic_and_no_verified_hardware_roofs'),
        hardware_headroom=_unsupported('no_verified_hardware_reference; hypothetical_roof_is_not_a_validated_limit'))


_LIMITATIONS=[
    'Analytic P12 FLOPs are operation estimates, not hardware counters; unknown operations keep total FLOPs null.',
    'Only the captured recipe/seed/input and executed paths are covered; no all-input model claim.',
    'Same captured forwards only. Independent P01/P02 latency/throughput is not silently merged across conditions.',
    'CPU and CUDA Event intervals include instrumentation; CUDA Event elapsed is not kernel-only compute time.',
    'Payload bytes assume every tensor argument occurrence is read and every output written once per dispatch.',
    'Aliases/views can be counted repeatedly; caching, fusion, workspaces and memory hierarchy are not observed.',
    'Estimated payload bytes are neither actual DRAM bytes nor a proven upper/lower bound.',
    'P15 H2D/D2H bandwidth is not GPU DRAM bandwidth; P10/P23 RSS, parameters and allocator bytes are not traffic.',
    'Theoretical references are caller supplied and unverified; no calibrated roof, bottleneck or headroom is claimed.',
    'Capture/observer overhead is included and not separately calibrated; no extra forward during analysis.',
    'No resource-occupancy admission gate or default RAM/VRAM quota; existing capture errors and OOM propagate.']


def analyze_roofline(source,*,protocol=RooflineProtocol(),reference=None):
    """Pure analysis of one existing P14/P12/P13 capture; no model execution.

    Default arithmetic intensity is explicitly estimated. Measured DRAM traffic,
    calibrated hardware roofs, bottleneck and valid hardware gap stay unsupported.
    """
    if type(protocol) is not RooflineProtocol:raise TypeError('RooflineProtocol required')
    if reference is not None and type(reference) is not RooflineReference:raise TypeError('RooflineReference required')
    data=_profile(source)
    result=_base('roofline','P18',dict(id='same_capture_roofline',version=1,scope='owned_profiled_inference',
        configuration=asdict(protocol)),data['provenance'])
    result['identity']['source_result_id']=data['result_id']
    result.update(requested=data['requested'],actual=data['actual'],measurements=_derive(data,protocol,reference),
        raw_evidence=dict(source=data,source_sha256=_hash(data),reference=asdict(reference) if reference else None),
        diagnostics=dict(warnings=[],limitations=_LIMITATIONS))
    return RooflineResult(result)


def measure_roofline(recipe,*,protocol=RooflineProtocol(),profiling_protocol=None):
    """One existing owned capture path; default one forward, no extra benchmark.

    Trusted importable recipe, eval/inference mode; no live caller model transfer.
    No external profiler install or hardware calibration stress is performed.
    """
    if type(recipe) is not ColdStartRecipe:raise TypeError('ColdStartRecipe required')
    if type(protocol) is not RooflineProtocol:raise TypeError('RooflineProtocol required')
    capture=ProfilingProtocol(warmup_iterations=0,measure_iterations=1) if profiling_protocol is None else profiling_protocol
    return analyze_roofline(profile_operators(recipe,protocol=capture),protocol=protocol)


def _validate_roofline(data):
    p=data['protocol'];raw=data['raw_evidence']
    if p.get('id')=='ncu_empirical_roofline':
        from .roofline_counters import _validate_counter_result
        _validate_counter_result(data)
        return
    if set(p)!={'id','version','scope','configuration'} or p['id']!='same_capture_roofline' or p['scope']!='owned_profiled_inference':
        raise ValueError('invalid P18 protocol/scope')
    if set(raw)!={'source','source_sha256','reference'}:raise ValueError('invalid P18 evidence fields')
    protocol=RooflineProtocol(**p['configuration'])
    source=OperatorProfileResult(raw['source']).to_dict()
    if raw['source_sha256']!=_hash(source) or data['identity']['source_result_id']!=source['result_id']:
        raise ValueError('P18 source association mismatch')
    if any(data[k]!=source[k] for k in ('provenance','requested','actual')):raise ValueError('P18 capture context mismatch')
    reference=RooflineReference(**raw['reference']) if raw['reference'] is not None else None
    if data['measurements']!=_derive(source,protocol,reference):raise ValueError('P18 derived evidence mismatch')
    if data['diagnostics']!=dict(warnings=[],limitations=_LIMITATIONS):raise ValueError('P18 limitations mismatch')
