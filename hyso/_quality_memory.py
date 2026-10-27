"""P29 memory-domain binding and observed Pareto arithmetic; no GPU fallback."""
import math

from .evaluation import _hash


MEMORY_METRICS = (None, 'logical_model_bytes', 'cpu_rss_sampled_peak_bytes',
                  'cuda_allocated_peak_bytes', 'cuda_reserved_peak_bytes', 'physical_vram_bytes')
_SCOPES = {
    None: 'not_requested',
    'logical_model_bytes': 'unique_registered_parameter_buffer_object_bytes',
    'cpu_rss_sampled_peak_bytes': 'current_process_sampled_peak_separate_single_forward',
    'cuda_allocated_peak_bytes': 'cuda_allocator_allocated_peak',
    'cuda_reserved_peak_bytes': 'cuda_allocator_reserved_peak',
    'physical_vram_bytes': 'physical_device_vram',
}


def binding(bundle, cohort):
    p = bundle['performance']
    return dict(benchmark_run_id=p['identity']['run_id'], model_sha256=p['model']['weights_sha256'],
        structure_sha256=_hash(p['model']['structural_descriptor']), cohort_sha256=_hash(cohort),
        input_sha256=p['input']['content_sha256'], device=p['environment']['device'],
        runtime={k:p['environment'][k] for k in ('python_version','torch_version','platform')},
        precision=p['configuration']['precision'], host_id=bundle['host_id'],
        protocol_sha256=_hash(dict(configuration=p['configuration'], timing={k:p['timing'][k]
            for k in ('timing_method','timing_scope','synchronization_policy')})))


def capture(bundle, cohort, model, x, metric, protocol):
    """Own precision copy only. RSS executes one separate, guarded CPU forward."""
    from .performance.memory import measure_footprint, measure_memory, MemoryProtocol
    from .benchmark.runner import _profile, _Snapshot, _output
    import torch
    record = dict(metric=metric, binding=binding(bundle,cohort), source=None,
                  completed_forwards=0, missing_reason=None)
    if metric not in ('logical_model_bytes','cpu_rss_sampled_peak_bytes'):
        record['missing_reason'] = ('not_requested' if metric is None else
            'P29_CPU_adapter_has_no_matching_'+metric+'_measurement')
        return record
    modules, entries = _profile(model, x.dtype, torch.device('cpu'))
    guard = _Snapshot(modules, entries, x)
    if metric == 'logical_model_bytes':
        record['source'] = measure_footprint(model).to_dict()
    else:
        def workload():
            with torch.inference_mode():
                output = model(x)
                _output(output,len(x))
                guard.check(full=True)
                record['completed_forwards'] += 1
        record['source'] = measure_memory(workload,protocol=protocol or MemoryProtocol()).to_dict()
    guard.check(full=True)
    return record


def value(bundle, cohort, metric):
    """Revalidate raw P10/P11; never trust a caller-written aggregate byte count."""
    from .performance.result import FootprintResult, MemoryResult
    record = bundle['memory']
    if (set(record) != {'metric','binding','source','completed_forwards','missing_reason'}
            or record['metric'] != metric or record['binding'] != binding(bundle,cohort)):
        raise ValueError('memory model/input/runtime/precision/protocol binding mismatch')
    if type(record['completed_forwards']) is not int or record['completed_forwards'] not in (0,1):
        raise ValueError('invalid memory forward accounting')
    result = dict(metric=metric,scope=_SCOPES[metric],unit='byte',value=None,
                  status='unsupported',reason=record['missing_reason'])
    source = record['source']
    if source is None:
        if not isinstance(record['missing_reason'],str) or not record['missing_reason'] or record['completed_forwards']:
            raise ValueError('missing memory requires reason and zero completions')
        return result
    if record['missing_reason'] is not None:
        raise ValueError('measured memory cannot also be missing')
    if metric == 'logical_model_bytes':
        source = FootprintResult(source).to_dict()
        if source['protocol']['byte_accounting']!='unique_tensor_object_logical':
            raise ValueError('unsupported logical byte accounting')
        if source['provenance']['model']['weights_sha256'] != bundle['benchmark_model_sha256']:
            raise ValueError('footprint model differs from quality/timing model')
        descriptor = source['provenance']['model']['structural_descriptor']
        descriptor = dict(descriptor,tensors=[{k:v for k,v in t.items() if k!='element_size_bytes'} for t in descriptor['tensors']])
        if _hash(descriptor) != _hash(bundle['performance']['model']['structural_descriptor']):
            raise ValueError('footprint structure differs from benchmark')
        seen = set(); expected = 0
        for row in bundle['state_witness']:
            if row['object_alias'] not in seen:
                seen.add(row['object_alias']); expected += len(bytes.fromhex(row['bytes_hex']))
        measurement = source['measurements']['parameter_buffer_bytes']
        if measurement['value'] != expected or record['completed_forwards']:
            raise ValueError('logical footprint differs from registered tensor witnesses')
        result.update(value=measurement['value'],status='derived',reason=None)
    elif metric == 'cpu_rss_sampled_peak_bytes':
        source = MemoryResult(source).to_dict()
        measurement = source['measurements']['sampled_peak_rss_bytes']
        measured = measurement['value'] is not None
        if (source['protocol']['workload'] != 'synchronous_callable'
                or record['completed_forwards'] != int(measured)):
            raise ValueError('RSS must cover one completed separate inference or be unsupported')
        result.update(value=measurement['value'],status='measured' if measured else 'unsupported',reason=measurement['reason'])
    else:
        raise ValueError('CPU comparison cannot bind CUDA allocator/physical VRAM evidence')
    environment = source['provenance']['environment']; p = bundle['performance']['environment']
    if (environment['device'] != p['device'] or any(environment[a]!=p[b] for a,b in
            (('python','python_version'),('torch','torch_version'),('platform_detail','platform')))):
        raise ValueError('memory device/runtime differs from benchmark')
    return result


def pareto(qa, la, ma, qb, lb, mb):
    """Pure reference: maximize quality, minimize latency and same-domain bytes."""
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in (qa,la,ma,qb,lb,mb)):
        raise ValueError('finite nonnegative comparable dimensions required')
    if qb>=qa and lb<=la and mb<=ma and (qb>qa or lb<la or mb<ma):return 'candidate'
    if qa>=qb and la<=lb and ma<=mb and (qa>qb or la<lb or ma<mb):return 'baseline'
    return 'neither'


def compare(a, b, cohort, policy, qa, la, qb, lb, previous):
    metric = policy['memory_metric']
    if metric not in MEMORY_METRICS:raise ValueError('unsupported P29 memory metric')
    threshold = policy['max_memory_regression']
    if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError('invalid memory regression fraction')
    left, right = value(a,cohort,metric), value(b,cohort,metric)
    if metric=='cpu_rss_sampled_peak_bytes':
        from dataclasses import asdict
        from .performance.memory import MemoryProtocol
        protocol=policy['memory_protocol']
        if protocol!=asdict(MemoryProtocol(**protocol)):raise ValueError('invalid memory protocol')
        for bundle in (a,b):
            source=bundle['memory']['source']
            if source is not None and source['protocol']['configuration']!=protocol:
                raise ValueError('memory sampling protocols differ')
    elif policy['memory_protocol'] is not None:
        raise ValueError('unexpected memory protocol for non-RSS metric')
    available = left['value'] is not None and right['value'] is not None
    ma,mb = left['value'],right['value']
    relative = (mb-ma)/ma if available and ma else None
    memory = dict(status='comparable' if available else 'unsupported',metric=metric,scope=_SCOPES[metric],unit='byte',
        baseline=left,candidate=right,delta_bytes=mb-ma if available else None,
        relative_change=relative,relative_status='defined' if relative is not None else
            'undefined_zero_baseline' if available else 'missing_measurement',
        regression=relative>threshold if relative is not None else None,
        uncertainty='Logical bytes are exact under the object convention; RSS is sampled process-wide, sequential and environment-dependent.')
    return dict(previous, comparison_version=2, quality_speed_pareto_dominates=previous['pareto_dominates'],
        pareto_dominates=pareto(qa,la,ma,qb,lb,mb) if available else None,
        pareto_dimensions=['quality','latency','memory'],pareto_status='observed' if available else 'unsupported_missing_memory',
        memory=memory,memory_regression=memory['regression'],winner=None,
        limitations=['Pareto is descriptive of the recorded protocol, not a guaranteed winner or significant speedup.',
            'Memory domains are never summed or substituted; RSS is not model-only memory or an absolute peak.',
            'Latency is measured, not P02 completion-window throughput. Memory observation is outside timing/evaluation.',
            'CUDA allocator/physical VRAM are unsupported by this CPU adapter; no GPU probe or fallback runs.'])
