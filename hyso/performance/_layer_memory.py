"""Allocation events attributed in one exported profiler clock, never peaks/RSS.

This observes PyTorch allocation requests/frees, including instrumentation.
It neither measures physical RAM/VRAM traffic nor establishes object lifetimes.
"""
import math


def memory_events(trace, budget):
    rows=[]
    for index,event in enumerate(trace.get('traceEvents',[])):
        if event.get('name')!='[memory]':continue
        if len(rows)>=budget:raise ValueError('memory event budget exceeded')
        args=event.get('args',{})
        rows.append(dict(trace_index=index,phase=event.get('ph'),start_us=event.get('ts'),
            pid=event.get('pid'),tid=event.get('tid'),bytes=args.get('Bytes'),
            device_type=args.get('Device Type'),device_id=args.get('Device Id')))
    return rows


def attribute_memory(events,trace,invocations,inventory,device):
    """Inclusive scopes and unique innermost owner; ambiguous events stay unknown."""
    spans={};seen=set();owners=[]
    for row in invocations:
        matches=[e for e in trace if e['category']=='user_annotation' and e['name']==row['annotation']]
        if len(matches)==1:
            e=matches[0]
            if all(type(e[k]) is int and e[k]>=0 for k in ('pid','tid')) and all(
                type(e[k]) in (int,float) and math.isfinite(e[k]) and e[k]>=0 for k in ('start_us','duration_us')):
                spans[row['invocation_id']]=e
    for event in events:
        index=event['trace_index']
        if type(index) is not int or index<0 or index in seen:raise ValueError('duplicate/invalid memory trace index')
        seen.add(index)
        if event['phase']!='i' or type(event['bytes']) is not int or not event['bytes']:
            raise ValueError('invalid profiler allocation event')
        if type(event['start_us']) not in (int,float) or not math.isfinite(event['start_us']) or event['start_us']<0:
            raise ValueError('invalid allocation timestamp')
        for k in ('pid','tid','device_type','device_id'):
            if type(event[k]) is not int:raise ValueError('invalid allocation identity')
        domain=('cpu' if event['device_type']==0 and event['device_id']==-1 else
                'cuda' if device.startswith('cuda:') and event['device_type']==1 and
                event['device_id']==int(device.split(':')[1]) else None)
        matches=[i for i,e in spans.items() if e['pid']==event['pid'] and e['tid']==event['tid'] and
                 e['start_us']<=event['start_us']<e['start_us']+e['duration_us']]
        # Candidates must form a single invocation ancestry chain, not overlapping siblings.
        chain=[]
        if matches and len(spans)==len(invocations):
            leaf=max(matches);current=leaf
            while current is not None:
                chain.append(current);current=invocations[current]['parent_invocation_id']
            if set(matches)!=set(chain) or domain is None:chain=[]
        owners.append(dict(trace_index=index,domain=domain,invocation_id=chain[0] if chain else None,
            inclusive_invocation_ids=chain,candidate_invocation_ids=matches,
            reason=None if chain else 'outside_missing_or_ambiguous_module_scope_or_device'))
    domains=['cpu']+(['cuda'] if device.startswith('cuda:') else [])
    def summary(indices,domain):
        values=[e['bytes'] for e,o in zip(events,owners) if o['domain']==domain and indices(o)]
        # Zeros mean no observed events in a verified annotation, not zero actual allocation.
        return dict(allocation_bytes=sum(v for v in values if v>0),
            deallocation_bytes=-sum(v for v in values if v<0),net_change_bytes=sum(values),event_count=len(values))
    rows=[]
    for row in invocations:
        i=row['invocation_id'];values={}
        for domain in domains:
            observed=any(o['domain']==domain for o in owners)
            valid=(len(spans)==len(invocations) and not any(
                i in o['candidate_invocation_ids'] and o['invocation_id'] is None and o['domain']==domain for o in owners))
            values[domain]=dict(status='measured' if observed and valid else 'unsupported',
                inclusive=summary(lambda o:i in o['inclusive_invocation_ids'],domain) if observed and valid else None,
                exclusive=summary(lambda o:o['invocation_id']==i,domain) if observed and valid else None,
                peak_bytes=None,retained_bytes=None,
                reason=None if observed and valid else 'no_observed_domain_events_or_verified_module_span')
        rows.append(dict(invocation_id=i,domains=values))
    modules=[]
    for module in inventory:
        calls=[r for r in rows if invocations[r['invocation_id']]['module_id']==module['module_id']]
        domains_data={}
        for domain in domains:
            ok=bool(calls) and all(c['domains'][domain]['status']=='measured' for c in calls)
            domains_data[domain]=dict(status='measured' if ok else 'unsupported',
                **{scope:{key:sum(c['domains'][domain][scope][key] for c in calls)
                    for key in ('allocation_bytes','deallocation_bytes','net_change_bytes','event_count')} if ok else None
                    for scope in ('inclusive','exclusive')},peak_bytes=None,retained_bytes=None)
        modules.append(dict(module_id=module['module_id'],domains=domains_data))
    return dict(status='partial' if any(o['invocation_id'] is not None for o in owners) else 'unsupported',
        scope='observed_PyTorch_allocation_requests_and_frees_on_annotation_host_thread; not_physical_memory',
        allocation_bytes=None,retained_bytes=None,peak_bytes=None,
        reason='observed_event_attribution_only; completeness_lifetimes_and_true_layer_peaks_unsupported',
        unassigned_event_count=sum(o['invocation_id'] is None for o in owners),
        assignments=owners,invocations=rows,modules=modules,
        limitations=['Positive events are observed allocation requests, not memory_allocated before/after differences.',
            'Net change is signed observed allocation minus frees; it is not retained/live tensor memory.',
            'Inclusive nested modules overlap; do not add inclusive module totals.',
            'CUDA allocation attribution is host-side, not GPU kernel execution or allocator reserved memory.',
            'Unobserved/native allocations, other threads, lifetime ownership and actual peaks are unsupported.',
            'Profiler memory tracing and observers alter allocations and scheduling; overhead is included.'])
