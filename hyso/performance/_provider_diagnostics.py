"""P35 evidence assessment, without executing a runtime or querying a device."""
from collections import Counter,defaultdict
import math

from .runtime_result import _digest,_hash,_int,_name,_number


def _capture_context(e,report,strict):
    c=report.get('capture_context')
    if c is None:
        if strict:raise ValueError('new provider capture missing context')
        return dict(value=None,reason='legacy_capture_has_no_feed_fingerprints_or_host_run_intervals')
    expected_clock=dict(name='perf_counter_ns',unit='ns',instance_id=e['token']+':perf_counter_ns')
    expected_profile=dict(name='ORT_JSON_profile_host_relative',unit='us',origin='runtime_private_uncorrelated_with_host_perf_counter')
    if c.get('host_clock')!=expected_clock or c.get('profile_clock')!=expected_profile:raise ValueError('provider clock domain mismatch')
    feeds=c.get('input_fingerprints')
    if type(feeds) is not dict or not feeds:raise ValueError('input fingerprint ledger required')
    for name,value in feeds.items():
        if type(value) is not dict:raise ValueError('invalid input fingerprint')
        _name(name);_name(value.get('dtype'));_digest(value.get('sha256'))
        if type(value['shape']) is not list:raise ValueError('input fingerprint shape required')
        for dim in value['shape']:_int(dim)
    if _hash(feeds)!=report['input_sha256'] or c['input_unchanged'] is not True:
        raise ValueError('provider input fingerprint mismatch or mutated input')
    _digest((report.get('source') or {}).get('module_sha256'))
    spans=c['run_intervals'];n=report['completed_runs']
    if type(spans) is not list or len(spans)!=n:raise ValueError('incomplete host run interval ledger')
    last=0
    for index,span in enumerate(spans):
        _int(span['index']);_int(span['start_ns'],last);_int(span['end_ns'],span['start_ns']+1)
        if span['index']!=index or span['phase']!=('warmup' if index<e['protocol']['warmup'] else 'measured'):
            raise ValueError('provider run interval sequence/phase mismatch')
        last=span['end_ns']
    return dict(value=c,reason=None)


def _assessment(e,analysis,*,strict=False):
    from .providers import _decode_model,_is_runtime_transfer
    report=e.get('report') or {}
    requested=report.get('requested_providers',e.get('providers'))
    if requested is None:requested=['CPUExecutionProvider']
    session=report.get('session_providers');available=report.get('available_providers')
    completed=analysis['status']!='unavailable' and report.get('status')=='completed'
    common=dict(backend=report.get('runtime'),requested_providers=requested,available_providers=available,
        enabled_session_providers=session,run_status=report.get('status'),controller_reason=e['controller']['reason'],
        completed_runs=report.get('completed_runs'),model_sha256=report.get('model_sha256'),input_sha256=report.get('input_sha256'),
        factory_source=report.get('source'),runtime_profile=dict(source='evidence.report.trace',
            trace_sha256=report.get('trace_sha256'),session_binding=report.get('profile_binding')),
        source_reference='evidence.report;hash_consistency_is_not_signed_authenticity',
        failure_phase=report.get('phase'),failure_category=analysis.get('failure_kind'),error_type=report.get('error_type'),
        static_assignment=dict(value=None,reason='complete_static_assignment_not_collected_by_this_adapter'),
        unsupported_operators=dict(value=None,reason='provider_capability_registry_or_structured_rejection_not_recorded'),
        physical_device=dict(value=None,requested=report.get('device_selection'),reason='ORT_profile_provider_is_not_physical_UUID_attestation'),
        clock=dict(name='ORT_JSON_profile_host_relative',unit='us',origin=None,
            reason='runtime_private_origin;no_cross_clock_correlation_or_GPU_kernel_clock'),
        execution_error_retry=dict(detected=None,reason='no_runtime_failure_retry_history;node_placement_is_not_recovery'),
        performance_impact=dict(status='needs_comparable_experiment' if analysis.get('unexpected_executions') else 'not_assessed',
            latency_delta_ms=None,percentage=None,reason='no_matching_counterfactual_baseline;host_Node_sums_are_not_latency_penalty',
            synchronization_wait_us=None,synchronization_reason='no_separate_wait_or_device_synchronization_measurement'))
    if not completed:
        return dict(common,status='unavailable',reason=analysis.get('reason') or 'partial_or_aborted_run_no_complete_profile',
            observed_execution_providers=[],findings=[],node_coverage=None,transfers=[],dataflow_provider_boundaries=[],
            same_node_provider_changes=[],operator_timings=[],model_run_windows=[],capture_context=None,
            partial_evidence=dict(completed_counter=report.get('completed_runs'),
                reason='a_partial_counter_or_exit_zero_does_not_establish_node_execution_or_unknown_tail'))
    context=_capture_context(e,report,strict)
    for values in (requested,available,session):
        if values is None:continue
        if type(values) is not list or len(set(values))!=len(values):raise ValueError('distinct ordered provider identities required')
        for value in values:_name(value)
    selected=[x for x in session if x in requested]
    priority=dict(requested_order=requested,enabled_order=session,requested_subsequence_preserved=selected==requested,
        status='matches' if selected==requested else 'configuration_order_difference',execution_claim=False)
    selection=report.get('device_selection')
    if selection is not None:
        if selection['gpu_uuid_request']!=e['recipe']['gpu_uuid']:raise ValueError('provider device selection mismatch')
        if selection['gpu_uuid_request'] is not None and selection['cuda_visible_devices']!=selection['gpu_uuid_request']:
            raise ValueError('provider CUDA visibility mismatch')
    configured=e.get('provider_options') or {};actual=report.get('provider_options') or {}
    for provider,options in configured.items():
        if 'device_id' in options and 'device_id' in actual.get(provider,{}):
            if str(options['device_id'])!=str(actual[provider]['device_id']):raise ValueError('resolved provider device ID mismatch')
    _,source=_decode_model(report['model_base64'])
    _,optimized=_decode_model(report['optimized_base64'],optimized=e.get('graph_analysis_version',1)==2)
    names=[n.name for n in source.graph.node];unique=bool(names) and all(names) and len(set(names))==len(names)
    optimized_names=Counter(n.name for n in optimized.graph.node)
    transfer_nodes={n.name:n for n in optimized.graph.node if _is_runtime_transfer(n) and optimized_names[n.name]==1}
    raw=[(index,x) for index,x in enumerate(report['trace']) if x.get('cat')=='Node' and x.get('name','').endswith('_kernel_time')]
    rows=[];seen=set();indices={};by_source=defaultdict(list);transfers=[];findings=[];transfer_indices=set()
    for (index,event),mapped in zip(raw,analysis['events']):
        args=event.get('args',{});stamp=event.get('ts');duration=event.get('dur');provider=mapped['provider']
        node_index=args.get('node_index');node_id=None
        if node_index is not None:
            if type(node_index) not in (int,str) or not str(node_index).isdigit():raise ValueError('invalid runtime node ID')
            node_id=int(node_index);_int(node_id)
            identity=(event['name'],args.get('op_name'))
            if node_id in indices and indices[node_id]!=identity:raise ValueError('runtime node ID assigned to different operators')
            indices[node_id]=identity
        if stamp is not None:_number(stamp)
        if duration is not None:_number(duration)
        signature=(event['pid'],event.get('tid'),stamp,duration,event['name'],provider,node_id)
        duplicate=stamp is not None and signature in seen
        if duplicate and strict:raise ValueError('duplicate provider profile execution event')
        if stamp is not None:seen.add(signature)
        complete_identity=stamp is not None and event.get('ph')=='X' and node_id is not None and not duplicate
        row=dict(mapped,raw_trace_index=index,runtime_node_id=node_id,profile_timestamp_us=stamp,
            event_identity_complete=complete_identity,evidence_source=f'evidence.report.trace[{index}]',
            kind='observed_runtime_profiler_event',clock='ORT_JSON_profile_host_relative_us')
        rows.append(row)
        if mapped['source_node'] is not None:by_source[mapped['source_node']].append(row)
        stem=event['name'][:-12];node=transfer_nodes.get(stem)
        if node is not None and args.get('op_name')==node.op_type:
            if complete_identity and provider is not None:transfer_indices.add(index)
            transfers.append(dict(optimized_node=stem,source_node=None,op_type=node.op_type,provider=provider,
                direction='device_to_host' if node.op_type=='MemcpyToHost' else 'host_to_device',
                host_node_duration_us=duration,duration_reason=None if duration is not None else 'profile_duration_missing',
                transferred_bytes=None,bytes_reason='DMA_byte_counter_not_recorded;activation_size_is_not_transfer_byte_evidence',
                device_dma_duration_us=None,dma_reason='ORT_host_Node_duration_is_not_device_DMA_timing',
                evidence_source=row['evidence_source'],kind='observed_runtime_transfer_operator',
                clock=row['clock'],timestamp_us=stamp,physical_device_id=None))
        if mapped['expected_provider'] is not None and provider is not None and provider!=mapped['expected_provider']:
            findings.append(dict(scope='observed_node_provider_placement',status='observed',
                expected_provider=mapped['expected_provider'],observed_provider=provider,node=mapped['source_node'],operator=args.get('op_name'),
                evidence_source=row['evidence_source'],kind='measured_profiler_placement_compared_with_declared_expectation',
                previous_provider=None,next_provider=None,transition_reason='expectation_is_not_previous_provider;no_same_node_transition_in_this_event',
                reliability='mapped_source_node_and_runtime_event;not_signed_authenticity',
                limitation='not_proof_of_unsupported_operator_or_runtime_failure_retry',
                next_validation='matching_counterfactual_run_only_if_causal_performance_impact_is_required'))
    counts={name:len(by_source[name]) for name in names if name}
    covered=unique and all(counts.get(name)==report['completed_runs'] and all(x['provider'] is not None and x['event_identity_complete'] for x in by_source[name]) for name in names)
    unknown_compute=[x for x in rows if x['source_node'] is None and x['raw_trace_index'] not in transfer_indices]
    covered=covered and not unknown_compute
    coverage=dict(status='complete' if covered else 'partial_or_unknown',source_node_count=len(names),
        source_names_unique=unique,events_per_source_node=counts,expected_events_per_source_node=report['completed_runs'],
        missing_or_partial_nodes=[name for name in names if counts.get(name)!=report['completed_runs'] or
            not all(x['provider'] is not None and x['event_identity_complete'] for x in by_source[name])],
        opaque_runtime_compute_events=len(unknown_compute),reason=None if covered else 'missing_duplicate_or_ambiguous_node_execution_identity;fusion_or_incomplete_coverage')
    # Dependency edges describe observed providers on graph producers/consumers.
    # They do not invent a DMA event for an uninstrumented edge.
    producer={output:n.name for n in source.graph.node for output in n.output}
    boundaries=[];history=[]
    for node in source.graph.node:
        if not unique:break
        providers={x['provider'] for x in by_source[node.name] if x['provider'] is not None}
        for value in node.input:
            previous=producer.get(value)
            if previous is None:continue
            before={x['provider'] for x in by_source[previous] if x['provider'] is not None}
            if len(before)==len(providers)==1 and before!=providers:
                boundaries.append(dict(producer_node=previous,consumer_node=node.name,tensor=value,
                    producer_provider=next(iter(before)),consumer_provider=next(iter(providers)),
                    kind='derived_graph_edge_between_observed_provider_events',
                    evidence_source='evidence.report.model_base64+mapped_profiler_events',
                    transferred_bytes=None,transfer_duration_us=None,
                    reason='graph_provider_boundary_alone_is_not_a_measured_physical_transfer'))
        ordered=sorted((x for x in by_source[node.name] if x['profile_timestamp_us'] is not None),key=lambda x:x['profile_timestamp_us'])
        for a,b in zip(ordered,ordered[1:]):
            if a['provider'] is not None and b['provider'] is not None and a['provider']!=b['provider']:
                history.append(dict(node=node.name,previous_provider=a['provider'],next_provider=b['provider'],
                    previous_evidence=a['evidence_source'],next_evidence=b['evidence_source'],
                    reason='observed_same_node_provider_change;cause_or_failure_retry_not_established'))
    groups={}
    for row in rows:
        key=(row.get('op_type') or report['trace'][row['raw_trace_index']].get('args',{}).get('op_name'),row['provider'])
        groups.setdefault(key,[]).append(row['duration_us'])
    timings=[]
    for (op,provider),values in groups.items():
        timed=[x for x in values if x is not None];complete=bool(timed) and len(timed)==len(values)
        timings.append(dict(operator=op,provider=provider,event_count=len(values),timed_event_count=len(timed),
            total_host_node_us=math.fsum(timed) if complete else None,known_duration_sum_us=math.fsum(timed) if timed else None,
            reason=None if complete else 'incomplete_event_durations;known_sum_is_not_total',
            kind='derived_descriptive_host_profiler_sum;overlap_possible_not_session_latency'))
    model_runs=[];complete_window_sources=set()
    for index,event in enumerate(report['trace']):
        if event.get('cat')!='Session' or event.get('name')!='model_run':continue
        if event.get('pid')!=e['controller']['launcher_pid']:raise ValueError('session event belongs to another worker')
        stamp=event.get('ts');duration=event.get('dur')
        if stamp is not None:_number(stamp)
        if duration is not None:_number(duration)
        if event.get('ph')=='X' and stamp is not None and duration is not None:
            complete_window_sources.add(f'evidence.report.trace[{index}]')
        model_runs.append(dict(timestamp_us=stamp,duration_us=duration,clock='ORT_JSON_profile_host_relative_us',
            source=f'evidence.report.trace[{index}]',scope='instrumented_ORT_session_model_run;not_a_counterfactual_latency_penalty'))
    if report['completed_runs']>1:
        # Equal aggregate counts can conceal all events belonging to one run.
        # Require a unique local profile window for each mapped source event.
        windows=[x for x in model_runs if x['source'] in complete_window_sources]
        windows.sort(key=lambda x:x['timestamp_us'])
        per_run=[Counter() for _ in windows]
        associated=len(windows)==report['completed_runs'] and all(
            a['timestamp_us']+a['duration_us']<=b['timestamp_us'] for a,b in zip(windows,windows[1:]))
        for row in rows:
            if row['source_node'] is None:continue
            stamp=row['profile_timestamp_us'];duration=row['duration_us'] or 0
            matches=[i for i,w in enumerate(windows) if stamp is not None and
                w['timestamp_us']<=stamp and stamp+duration<=w['timestamp_us']+w['duration_us']]
            if len(matches)!=1:associated=False
            else:per_run[matches[0]][row['source_node']]+=1
        associated=associated and all(all(run.get(name)==1 for name in names) for run in per_run)
        coverage['per_run_source_event_counts']=[dict(run) for run in per_run]
        coverage['per_run_association']='complete' if associated else 'unknown_or_incomplete'
        if not associated:
            covered=False;coverage.update(status='partial_or_unknown',reason='aggregate_counts_do_not_prove_each_profile_run_is_covered')
    return dict(common,status='observed' if rows else 'unknown',reason=None if rows else 'no_node_execution_events',
        provider_priority=priority,observed_execution_providers=sorted({x['provider'] for x in rows if x['provider'] is not None}),
        node_coverage=coverage,events=rows,findings=findings,transfers=transfers,dataflow_provider_boundaries=boundaries,
        same_node_provider_changes=history,operator_timings=timings,model_run_windows=model_runs,capture_context=context,
        explicit_CPU_execution=bool(rows) and requested==['CPUExecutionProvider'] and all(x['provider']=='CPUExecutionProvider' for x in rows),
        placement_mismatch=dict(detected=True if findings else False if covered and e['expected'] else None,
            reason=None if findings or (covered and e['expected']) else 'no_complete_mapping_and_declared_expectations',
            scope='unexpected_observed_placement;not_error_retry_or_proven_unsupported_kernel'))
