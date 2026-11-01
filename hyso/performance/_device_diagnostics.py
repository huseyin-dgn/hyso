"""P34 descriptive comparisons. All analysis is offline and duration based.

Old P34 envelopes keep their exact derivation. New captures attest timestamps
and factory provenance; replay never reads the original checkpoint or source.
"""
import hashlib
import math
from pathlib import Path

from .runtime_result import _digest, _hash, _int, _name
from .result import _quantile


def _file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()


def _state_hash(model):
    from .devices import _tensor_hash
    return _hash({key:_tensor_hash(value) for key,value in model.state_dict().items()})


def _checkpoint_unchanged(origin):
    checkpoint=origin['checkpoint']
    if checkpoint is not None and _file_hash(checkpoint['path'])!=checkpoint['sha256']:
        raise ValueError('checkpoint fingerprint changed')


def _capture_origin(declared,model,source):
    """An optional third factory result declares origin; observed hashes bind it.

    Trusted executable factory declarations are not signed proof that training
    happened or that a preprocessing implementation matches its description.
    """
    _digest(source['module_sha256'])
    loaded=_state_hash(model)
    if declared is None:
        return dict(kind='factory_generated_no_checkpoint_claim',model_id=type(model).__module__+'.'+type(model).__qualname__,
            checkpoint=None,checkpoint_reason='factory_did_not_declare_checkpoint_origin',loaded_state_sha256=loaded,
            preprocessing_id='factory_source_and_configuration',
            preprocessing_sha256=_hash(dict(module_sha256=source['module_sha256'],configuration=source['configuration'])),
            preprocessing_identity_source='observed_factory_source_and_configuration;input_content_attested_separately')
    if type(declared) is not dict or declared.get('kind') not in ('trained_checkpoint','controlled_fixture'):
        raise ValueError('explicit factory origin kind required')
    for key in ('model_id','preprocessing_id'):_name(declared[key])
    _digest(declared['preprocessing_sha256'])
    if declared.get('expected_loaded_state_sha256') is not None:
        _digest(declared['expected_loaded_state_sha256'])
        if declared['expected_loaded_state_sha256']!=loaded:raise ValueError('loaded state fingerprint mismatch')
    checkpoint=None
    if declared['kind']=='trained_checkpoint':
        _digest(declared['checkpoint_sha256']);_name(declared['checkpoint_path'])
        if not Path(declared['checkpoint_path']).is_absolute():raise ValueError('absolute checkpoint path required')
        checkpoint=dict(path=declared['checkpoint_path'],sha256=declared['checkpoint_sha256'])
    result=dict(kind=declared['kind'],model_id=declared['model_id'],checkpoint=checkpoint,
        checkpoint_reason=None if checkpoint else 'controlled_fixture_has_no_trained_checkpoint',loaded_state_sha256=loaded,
        preprocessing_id=declared['preprocessing_id'],preprocessing_sha256=declared['preprocessing_sha256'],
        preprocessing_identity_source='trusted_factory_declaration;observed_source_and_input_recorded_separately')
    _checkpoint_unchanged(result)
    return result


def _origin(ident,run,modern):
    value=ident.get('origin')
    if value is None:
        if modern:raise ValueError('new capture missing workload origin')
        # Only a code/config/input identity proxy is recoverable from old files.
        return dict(kind='legacy_origin_not_recorded',model_id=None,checkpoint=None,
            checkpoint_reason='checkpoint_origin_not_recorded',loaded_state_sha256=None,
            preprocessing_id='legacy_factory_source_and_configuration_proxy',
            preprocessing_sha256=_hash(dict(module_sha256=ident['factory_sha256'],configuration=run['recipe']['configuration'])),
            preprocessing_identity_source='legacy_proxy_not_independent_preprocessing_attestation')
    if value['kind'] not in ('trained_checkpoint','controlled_fixture','factory_generated_no_checkpoint_claim'):
        raise ValueError('unknown workload origin kind')
    for key in ('model_id','preprocessing_id','preprocessing_identity_source'):_name(value[key])
    for key in ('loaded_state_sha256','preprocessing_sha256'):_digest(value[key])
    checkpoint=value['checkpoint']
    if value['kind']=='trained_checkpoint':
        if type(checkpoint) is not dict:raise ValueError('trained origin requires checkpoint evidence')
        _digest(checkpoint['sha256']);_name(checkpoint['path'])
        if value['checkpoint_reason'] is not None:raise ValueError('contradictory checkpoint status')
    else:
        if checkpoint is not None:raise ValueError('non-trained origin cannot attest trained checkpoint')
        _name(value['checkpoint_reason'])
    return value


def _timing(report,ident,modern):
    window=report.get('measurement_window');intervals=report.get('sample_intervals')
    if window is None and intervals is None and not modern:
        return dict(value=None,reason='legacy_capture_has_duration_only_no_absolute_boundaries')
    if type(window) is not dict or type(intervals) is not list:raise ValueError('window and sample intervals required')
    if window['clock']!=ident['clock'] or window['clock_instance_id']!=str(ident['worker_pid'])+':perf_counter_ns':
        raise ValueError('measurement clock domain mismatch')
    _int(window['start_ns']);_int(window['end_ns'],window['start_ns']+1)
    if window['end_ns']-window['start_ns']!=report['window_ns']:raise ValueError('observation duration mismatch')
    if len(intervals)!=len(report['latency_ns']):raise ValueError('sample interval count mismatch')
    last=window['start_ns']
    for index,(sample,latency) in enumerate(zip(intervals,report['latency_ns'])):
        _int(sample['index']);_int(sample['start_ns'],last);_int(sample['end_ns'],sample['start_ns']+1,window['end_ns'])
        if sample['index']!=index or sample['end_ns']-sample['start_ns']!=latency:
            raise ValueError('sample sequence/duration mismatch')
        last=sample['end_ns']
    return dict(value=window,reason=None)


def _metric(value,unit,method,reason=None):
    return dict(value=value,unit=unit,kind='derived',status='derived' if value is not None else 'unsupported',
        reason=None if value is not None else reason or 'not_comparable',method=method)


def _diagnostics(e,base,*,baseline_device=None):
    modern=e.get('comparison_version')==3
    names=e['devices']
    baseline_device=baseline_device if baseline_device is not None else ('cpu' if 'cpu' in names else names[0])
    if baseline_device not in names:raise ValueError('baseline must name a recorded device')
    rows=[];contracts={}
    for run,row in zip(e['runs'],base['devices']):
        device=row['device'];report=run['report']
        source=dict(worker_token=run['token'],request_sha256=report.get('request_sha256') if report else None,
            report_sha256=_hash(report),controller_sha256=_hash(run['controller']))
        if row['status']!='measured':
            rows.append(dict(device=device,status=row['status'],reason=row['reason'],source=source,
                contract=None,metrics=None,trained_checkpoint_attested=False));continue
        ident=row['identity'];samples=report['latency_ns'];n=len(samples)
        if modern and report['loaded_state_unchanged'] is not True:raise ValueError('inference mutated loaded state')
        origin=_origin(ident,run,modern);timing=_timing(report,ident,modern)
        execution=ident.get('execution')
        contract=dict(model=ident['model_sha256'],input=ident['input_sha256'],factory=ident['factory_sha256'],
            factory_entrypoint=run['recipe']['factory'],configuration=run['recipe']['configuration'],shape=ident['shape'],batch=ident['batch'],
            precision=ident['precision'],backend=ident['backend'],runtime=ident['runtime'],
            warmup=ident['warmup'],samples=ident['measured_samples'],seed=run['protocol']['seed'],
            settings=execution['settings'] if execution else None,output_shape=execution.get('output_shape') if execution else None,
            clock_method=ident['clock'],latency_scope=ident['scope'],
            throughput_scope=row['throughput']['scope'],
            transfers='initial_H2D_and_D2H_excluded;resident_input_output;validation_outside_latency_inside_throughput_window',
            cold_start='not_measured;setup_and_warmup_excluded;steady_state_not_proven',
            origin_kind=origin['kind'],model_id=origin['model_id'],loaded_state_sha256=origin['loaded_state_sha256'],
            checkpoint_sha256=origin['checkpoint']['sha256'] if origin['checkpoint'] else None,
            preprocessing_id=origin['preprocessing_id'],preprocessing_sha256=origin['preprocessing_sha256'])
        contracts[device]=contract
        quantiles={}
        for label,q in (('p50',.5),('p95',.95),('p99',.99)):
            sufficient=q==.5 or n>=1000
            quantiles[label]=_metric(_quantile(samples,q)/1e6 if sufficient else None,'ms/batch',
                'linear_order_statistic;P95_P99_require_1000_samples',None if sufficient else 'insufficient_tail_samples_minimum_1000')
        metrics=dict(mean_batch_latency=_metric(math.fsum(samples)/n/1e6,'ms/batch','arithmetic_mean_of_completed_forward_latencies'),
            **quantiles,completed_forward_throughput=_metric(n/(report['window_ns']/1e9),'forward/s','successful_forward_count/instrumented_completion_window_seconds'),
            images_per_second=_metric(report['completed']/(report['window_ns']/1e9),'image/s','completed_image_count/instrumented_completion_window_seconds'))
        rows.append(dict(device=device,status='measured',reason=None,source=source,origin=origin,contract=contract,metrics=metrics,
            trained_checkpoint_attested=origin['kind']=='trained_checkpoint',
            sample_count=n,completed_forwards=n,completed_images=report['completed'],duration_seconds=report['window_ns']/1e9,
            timing=timing,synchronization=ident.get('synchronization'),device_name=ident.get('device_name'),
            device_hardware=ident.get('hardware'),memory=row['memory'],
            process_private_commit=dict(value=None,unit='byte',domain='process_private_commit_snapshot',
                reason='P34_capture_does_not_record_private_commit;RSS_is_not_a_substitute'),
            missing_sensors=dict(energy=row['energy'],quality=row['quality'])))
    baseline=next(row for row in rows if row['device']==baseline_device);pairs=[]
    for candidate in rows:
        if candidate is baseline:continue
        ready=baseline['contract'] is not None and candidate['contract'] is not None
        differences=[];matching=[]
        if ready:
            for key,value in baseline['contract'].items():
                (matching if value==candidate['contract'][key] else differences).append(key)
        # Legacy v1 does not attest actual residency/settings/physical mapping.
        legacy_unverified=e.get('comparison_version',1)==1
        comparable=ready and not differences and not legacy_unverified
        reason=None if comparable else ('unsuccessful_or_partial_device_run' if not ready else
            'legacy_execution_mapping_not_attested' if legacy_unverified else 'execution_contract_mismatch:'+','.join(differences))
        numbers=dict(latency_difference_ms=None,latency_difference_percent=None,latency_speedup=None,throughput_ratio=None)
        if comparable:
            b=baseline['metrics'];c=candidate['metrics'];bl=b['mean_batch_latency']['value'];cl=c['mean_batch_latency']['value']
            numbers=dict(latency_difference_ms=cl-bl,latency_difference_percent=100*(cl-bl)/bl,
                latency_speedup=bl/cl,throughput_ratio=c['images_per_second']['value']/b['images_per_second']['value'])
        memory={}
        from .devices import _MEMORY_DOMAINS
        for key,scope in _MEMORY_DOMAINS.items():
            def observed(row):
                if row['contract'] is None:return None,None
                return row['memory']['value'].get(key),row['memory'].get('domains',{}).get(key)
            bv,bd=observed(baseline);cv,cd=observed(candidate)
            eligible=comparable and key=='logical_model_bytes' and bv is not None and cv is not None and bd==cd==dict(unit='byte',scope=scope)
            memory[key]=dict(baseline_value=bv,candidate_value=cv,baseline_domain=bd,candidate_domain=cd,
                comparable=eligible,difference_bytes=cv-bv if eligible else None,
                reason=None if eligible else 'missing_or_incompatible_domain_scope_or_whole_worker_not_model_memory')
        trained=comparable and baseline['trained_checkpoint_attested'] and candidate['trained_checkpoint_attested']
        pairs.append(dict(baseline_device=baseline_device,candidate_device=candidate['device'],
            status='comparable' if comparable else 'not_comparable',reason=reason,
            matching_conditions=matching,differences=differences,
            latency_difference=_metric(numbers['latency_difference_ms'],'ms/batch','candidate_minus_baseline',reason),
            latency_difference_percent=_metric(numbers['latency_difference_percent'],'percent','100*(candidate-baseline)/baseline',reason),
            latency_speedup=_metric(numbers['latency_speedup'],'ratio','baseline_mean_latency/candidate_mean_latency;above_1_means_candidate_faster',reason),
            throughput_ratio=_metric(numbers['throughput_ratio'],'ratio','candidate_images_per_second/baseline_images_per_second;above_1_means_candidate_faster',reason),
            memory=memory,trained_model_comparison=dict(comparable=trained,
                reason=None if trained else reason or 'no_matching_trained_checkpoint_origin_evidence'),ranking=None))
    comparable=bool(pairs) and all(p['status']=='comparable' for p in pairs)
    return dict(status='comparable' if comparable else 'not_comparable',comparable=comparable,
        baseline_device=baseline_device,baseline_selection='explicit_argument_or_CPU_else_first_recorded_device',
        devices=rows,comparisons=pairs,ranking=None,clock_policy='compare_local_durations_only;never_subtract_cross_process_absolute_timestamps',
        uncertainty=dict(repeated_run_distribution=None,confidence_interval=None,
            reason='one_serial_capture_per_device;samples_not_proven_independent;thermal_and_environment_variation_unquantified'),
        provenance_limit='hashes_detect_inconsistency_not_signed_authenticity;factory_declarations_are_trusted_code_claims',
        throughput_latency_note='throughput_uses_instrumented_completion_wall;latency_excludes_validation_heartbeat_and_pre_drain;neither_is_inverse_of_the_other')


def _analyze_v3(e):
    from .devices import _analyze_devices_v2
    base=_analyze_devices_v2(e)
    diagnostics=_diagnostics(e,base)
    comparable=diagnostics['comparable']
    base['comparable']=comparable
    base['status']='comparable' if comparable else 'limited'
    if not comparable:base['reason']='requires_successful_matching_execution_and_origin_contracts_on_distinct_devices'
    for key in ('latency','throughput'):
        base['axes'][key]=dict(comparable=comparable,reason=None if comparable else base['reason'])
    if not comparable:
        for value in base['axes'].values():
            if value['comparable']:value.update(comparable=False,reason=base['reason'])
    base['differences']=sorted(set(base['differences'])|{key for pair in diagnostics['comparisons'] for key in pair['differences']})
    base['diagnostics']=diagnostics
    return base
