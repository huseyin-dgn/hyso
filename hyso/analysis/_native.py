"""Closed native-reader adapters. Never import a source-declared factory."""
from ._codec import EvidenceError, plain, digest

_PERFORMANCE = {
    'latency_analysis': 'LatencyAnalysis', 'throughput': 'ThroughputResult',
    'memory': 'MemoryResult', 'footprint': 'FootprintResult', 'pipeline': 'PipelineResult',
    'stage_timing': 'StageTimingResult', 'cold_start': 'ColdStartResult',
    'sweep': 'SweepResult', 'precision': 'PrecisionResult', 'capacity': 'CapacityResult',
    'operator_profile': 'OperatorProfileResult', 'layer_profile': 'LayerProfileResult',
    'operation_count': 'OperationCountResult', 'transfer': 'TransferResult',
    'utilization': 'UtilizationResult', 'roofline': 'RooflineResult',
    'layer_overhead': 'LayerOverheadResult',
}


def validate_native(payload):
    """Reuse native pure constructors, including their original version rules."""
    d = plain(payload)
    if type(d) is not dict or type(d.get('schema_version')) is not int or d['schema_version'] != 1:
        raise EvidenceError('unsupported_source_version')
    fmt = d.get('format')
    try:
        if fmt is None and d.get('benchmark_type') == 'inference':
            from hyso.benchmark import BenchmarkResult
            cls = BenchmarkResult
            features = ['P01', 'P09']
        elif fmt and fmt.startswith('hyso.performance.') and fmt[17:] in _PERFORMANCE:
            from hyso.performance import result
            cls = getattr(result, _PERFORMANCE[fmt[17:]])
            features = d.get('feature_ids')
        elif fmt == 'hyso.evaluation.classification':
            from hyso.evaluation import EvaluationResult
            cls, features = EvaluationResult, ['P26']
        elif fmt == 'hyso.quality_experiment':
            from hyso.quality import QualityExperimentResult
            cls, features = QualityExperimentResult, [d.get('feature_id')]
        elif fmt == 'hyso.performance.runtime':
            from hyso.performance.runtime_result import RuntimeResult
            cls, features = RuntimeResult, [d.get('feature_id')]
        else:
            from hyso.performance.training import TrainingPerformanceResult
            from hyso.performance.training_details import TrainingStageResult, TrainingThroughputResult
            from hyso.performance.dataloader import DataLoaderAnalysisResult
            from hyso.performance.training_memory import TrainingMemoryResult
            from hyso.performance.gradients import GradientDiagnosticsResult
            readers = {
                'hyso.performance.training_timing': TrainingPerformanceResult,
                'hyso.performance.training_stages': TrainingStageResult,
                'hyso.performance.training_throughput': TrainingThroughputResult,
                'hyso.performance.dataloader': DataLoaderAnalysisResult,
                'hyso.performance.training_memory': TrainingMemoryResult,
                'hyso.performance.gradient_diagnostics': GradientDiagnosticsResult,
            }
            if fmt not in readers:
                raise EvidenceError('unsupported_source_format')
            cls, features = readers[fmt], d.get('feature_ids')
        cls(d)  # Constructor verifies raw/derived consistency; no measure call.
    except EvidenceError:
        raise
    except ImportError as exc:
        raise EvidenceError('native_reader_unavailable') from exc
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError, OverflowError, AttributeError) as exc:
        raise EvidenceError('invalid_native_evidence') from exc
    if not features or any(f not in {'P%02d' % i for i in range(1, 36)} for f in features):
        raise EvidenceError('invalid_feature_identity')
    return cls.__module__ + '.' + cls.__name__, features


def at(data, path, default=None):
    current = data
    try:
        for key in path.split('.'):
            current = current[int(key)] if isinstance(current, list) else current[key]
        return current
    except (KeyError, IndexError, TypeError, ValueError):
        return default


def native_scope(d, feature):
    source = d.get('source_p19', d)
    p = source.get('provenance', {})
    protocol = d.get('protocol', d.get('configuration', {}))
    if d.get('benchmark_type') == 'inference':
        p=dict(model=d['model'],input=d['input'],environment=d['environment'])
    if d.get('format') == 'hyso.performance.runtime':
        protocol = at(d,'evidence.protocol',at(d,'evidence.request.protocol',{}))
    input_data = p.get('input', {})
    env = p.get('environment', p.get('runtime', {}))
    model = p.get('model', {})
    shape = input_data.get('shape', input_data.get('input_shape'))
    device = env.get('device', at(protocol, 'configuration.device', at(protocol, 'recipe.device')))
    precision = at(protocol, 'configuration.precision', at(protocol, 'recipe.precision'))
    if feature == 'P26':
        device = at(d, 'source.device')
    if feature in ('P19', 'P20', 'P21', 'P22', 'P23', 'P25'):
        phase = 'training'
    elif feature in ('P24', 'P26', 'P27', 'P28', 'P29'):
        phase = 'evaluation'
    elif feature in ('P01','P02','P03','P04','P05','P06','P07','P08','P12','P13','P14','P17'):
        phase = 'inference'
    else:
        phase = None
    # No model-state hash is converted into a checkpoint file hash. No dataset
    # identity is guessed from the class name or shape.
    values = dict(device=device, precision=precision, input_shape=shape,
                  batch=shape[0] if shape else None, phase=phase,
                  model_state_sha256=model.get('weights_sha256', model.get('state_sha256')),
                  protocol_id=protocol.get('id'), protocol_sha256=digest(dict(
                      protocol=protocol,configuration=d.get('configuration'),
                      training=p.get('training'),optimizer=p.get('optimizer'))) if protocol else None,
                  clock_id=protocol.get('clock_id'), split=at(d, 'configuration.split'),
                  backend=env.get('backend') if type(env.get('backend')) is str else None, input_kind=None)
    if feature in ('P05', 'P12', 'P13', 'P14', 'P17') and (protocol.get('recipe') or feature in ('P12','P13','P14')):
        values['input_kind'] = 'synthetic_recipe'
    return {k: v for k, v in values.items() if v is not None}


_DOMAINS = {'P01':'latency','P02':'throughput','P03':'e2e','P04':'stage_timing',
 'P05':'cold_start','P06':'latency','P07':'latency','P08':'latency','P09':'cuda_allocator',
 'P10':'process_rss','P11':'model_logical_bytes','P12':'logical_operations',
 'P13':'layer_profile','P14':'operator_profile','P15':'transfer','P16':'utilization',
 'P17':'capacity','P18':'roofline_estimate','P19':'training_timing','P20':'training_stages',
 'P21':'loader_wait','P22':'training_throughput','P23':'training_memory','P24':'time_to_quality',
 'P25':'gradient','P26':'quality','P27':'quality','P28':'quality','P29':'observed_comparison',
 'P30':'streaming','P31':'stability','P32':'process_rss_trend','P33':'energy',
 'P34':'device_comparison','P35':'provider_execution'}


def metrics(d, feature):
    """Small projections with explicit units; retained native payload is canonical."""
    out = []
    def add(path, unit=None, status='derived', domain=None, reason=None):
        value = at(d, path)
        if type(value) not in (int, float) and value is not None:
            return
        out.append(dict(path=path, value=value, unit=unit,
                        status=status if value is not None or status in ('unknown','unsupported','not_run') else 'unknown',
                        reason=reason if value is not None else reason or 'not_available_in_source',
                        domain=domain or _DOMAINS[feature]))
    def wrappers(node, prefix):
        if type(node) is dict and 'value' in node and 'unit' in node:
            path = prefix + '.value'
            status = node.get('status')
            status = status if status in ('measured','derived','estimated','unsupported') else 'unknown'
            domain = _DOMAINS[feature]
            if 'rss' in prefix or 'hwm' in prefix: domain = 'process_rss'
            elif 'artifact' in prefix: domain = 'serialized_artifact_bytes'
            elif 'allocated' in prefix or 'reserved' in prefix: domain = 'cuda_allocator'
            elif 'parameter' in prefix or 'buffer' in prefix: domain = 'model_logical_bytes'
            add(path, node['unit'], status, domain, node.get('reason'))
        elif type(node) is dict:
            for key, value in node.items(): wrappers(value, prefix + '.' + key)
        elif type(node) is list:
            for i, value in enumerate(node): wrappers(value, prefix + '.' + str(i))
    wrappers(d.get('measurements', {}), 'measurements')
    if d.get('benchmark_type') == 'inference':
        if feature=='P01':
            for key in ('mean_ms','p50_ms','p95_ms','std_ms'):
                if key in d['stats']['pooled']:add('stats.pooled.'+key,'ms')
        if feature=='P09':
            for key in ('peak_allocated_bytes','peak_reserved_bytes'):
                add('cuda_memory.'+key,'byte','measured','cuda_allocator')
    if feature == 'P04':
        for stage in d['measurements']: add('measurements.'+stage+'.host_mean_ms','ms','derived')
    if feature in ('P06','P07','P08'):
        for i, point in enumerate(d['raw_evidence']['points']):
            wrappers((point.get('result') or {}).get('measurements', {}), 'raw_evidence.points.%d.result.measurements' % i)
    if feature == 'P12':
        for name, unit in [('total_macs_per_batch','MAC/batch'),('total_flops_per_batch','logical_FLOP/batch')]: add('measurements.'+name,unit)
    if feature == 'P13':
        for i, _ in enumerate(d['measurements']['modules']): add('measurements.modules.%d.cpu_total_us'%i,'us')
    if feature == 'P14':
        for i, _ in enumerate(d['measurements']['operators']): add('measurements.operators.%d.self_cpu_us'%i,'us')
    if feature == 'P16': add('measurements.cpu.machine_capacity_percent','percent','measured')
    if feature == 'P17': add('measurements.maximum_observed_feasible_batch','images','derived'); add('measurements.maximum_proven_within_range','images')
    if feature in ('P19', 'P22'):
        for i, _ in enumerate(d.get('epochs', [])):
            add('epochs.%d.training_s'%i,'s','measured')
        if feature == 'P22': wrappers(d['segment']['rates'], 'segment.rates')
    if feature == 'P20':
        for i, _ in enumerate(d['summary']): add('summary.%d.host_ms'%i,'ms')
    if feature == 'P21':
        for i, _ in enumerate(d['summaries']): add('summaries.%d.next_wait_ms'%i,'ms')
    if feature == 'P23':
        wrappers(d['rss']['summary'], 'rss.summary')
        for i, row in enumerate(d['observations']):
            for component in ('parameter','buffer','gradient','optimizer'):
                add('observations.%d.logical.summary.%s.logical_bytes'%(i,component),'byte','derived','training_state_logical_bytes')
    if feature == 'P25':
        for i, _ in enumerate(d['observations']): add('observations.%d.preclip.summary.global_l2.value'%i,'L2','derived')
    if feature == 'P26':
        add('metrics.accuracy','fraction')
        for kind in ('macro','micro','weighted'):
            for name in ('precision','recall','f1'): add('metrics.averages.'+kind+'.'+name,'fraction')
    if feature in ('P27','P28'):
        for metric in ('accuracy','macro_f1'):
            for name in ('baseline','candidate','percentage_points','relative_percent'):
                add('analysis.'+metric+'.'+name,'percentage_point' if name=='percentage_points' else 'percent' if name=='relative_percent' else 'fraction')
    if feature == 'P29':
        add('analysis.quality.baseline','fraction'); add('analysis.quality.candidate','fraction')
        add('analysis.latency_relative_change','fraction',domain='observed_latency_change')
    if feature in ('P30','P31'):
        add('analysis.duration_seconds','s','measured'); add('analysis.completed','images','measured')
    if feature == 'P32': add('analysis.rss.slope_bytes_per_second','byte/s')
    if feature == 'P33':
        wrappers(d['analysis'],'analysis')
    if feature == 'P34':
        for i, _ in enumerate(d['analysis'].get('devices', [])):
            add('analysis.devices.%d.latency.value.mean_ms'%i,'ms')
    if feature == 'P35':
        for i, _ in enumerate(d['analysis'].get('events', [])): add('analysis.events.%d.duration_us'%i,'us','measured')
    return out


def native_outcome(d):
    for node in (d,d.get('analysis'),d.get('measurements')):
        if type(node) is dict and type(node.get('status')) is str:return node['status']
    return None


def limitations(feature, d):
    notes = ['Native constructor validation proves internal consistency, not capture authenticity or an undeclared dataset/checkpoint association.']
    status=native_outcome(d)
    if status is not None:notes.append('Native result status: '+status+'. Internal validity is distinct from measurement coverage or acceptance.')
    if feature == 'P18': notes += ['A derived/partial Roofline result is not a qualified stable roof. Reference qualification remains experiment-specific and is not certified here.']
    if feature == 'P29': notes += ['Observed Pareto/delta is not independently verified improvement; no recommendation or speedup certification is produced.']
    if feature == 'P31': notes += ['A short completed window is not the required 7200-second acceptance. Long-run acceptance remains PARTIAL.']
    if feature == 'P32': notes += ['RSS change/slope does not prove a memory leak.']
    if feature == 'P35': notes += ['Provider availability is not node execution; source/optimized mapping can remain unknown.']
    return notes
