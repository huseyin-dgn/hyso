"""Bounded plain-data profiler evidence. ID domains never cross captures."""
import math


def duration(value):
    return value if type(value) in (int,float) and math.isfinite(value) and value>=0 else None


def _id(value):
    return type(value) is int and value>=0


def validate_events(events):
    by_id={}
    for e in events:
        key=(e['device_type'],e['event_id'])
        if not _id(e['event_id']) or key in by_id:raise ValueError('duplicate/invalid profiler event ID')
        by_id[key]=e
        for k in ('cpu_total_us','self_cpu_us'):
            if e[k] is not None and duration(e[k]) is None:raise ValueError('invalid CPU duration')
        expected='measured' if e['cpu_total_us'] is not None and e['self_cpu_us'] is not None else 'invalid_or_missing'
        if e['timing_status']!=expected:raise ValueError('invalid CPU timing status')
        if e['device_type']=='CPU' and e['cpu_total_us'] is not None and e['self_cpu_us'] is not None:
            if e['self_cpu_us']>e['cpu_total_us']+1e-6:raise ValueError('self CPU duration exceeds total')
        if 'start_us' in e:
            if not _id(e['thread_id']):raise ValueError('invalid CPU thread ID')
            start=e['start_us'];end=e['end_us']
            if start is not None and duration(start) is None or end is not None and duration(end) is None:
                raise ValueError('invalid event interval')
            if start is not None and end is not None and end<start:raise ValueError('reversed event interval')
    for e in events:
        seen={e['event_id']};parent=e['parent_id']
        while parent is not None:
            p=by_id.get(('CPU',parent))
            if not _id(parent) or p is None or parent in seen:raise ValueError('invalid/cyclic CPU event parent')
            seen.add(parent);parent=p['parent_id']
    return by_id


def validate_modules(inventory,rows,forwards):
    modules={};aliases=set()
    for m in inventory:
        if m['module_id'] in modules:raise ValueError('duplicate module identity')
        if m['path'] not in m['aliases'] or len(set(m['aliases']))!=len(m['aliases']):raise ValueError('invalid aliases')
        if aliases.intersection(m['aliases']):raise ValueError('ambiguous module alias')
        aliases.update(m['aliases']);modules[m['module_id']]=m
    roots=[m for m in inventory if m['path']=='']
    if len(roots)!=1:raise ValueError('invalid module inventory root')
    counts=[0]*forwards;annotations=set()
    for i,r in enumerate(rows):
        index=r['measurement_index'];parent=r['parent_invocation_id']
        if type(r['invocation_id']) is not int or r['invocation_id']!=i:raise ValueError('invalid invocation ID')
        if type(index) is not int or not 0<=index<forwards or r['module_id'] not in modules:raise ValueError('invalid module association')
        for key in ('cpu_total_us','cpu_exclusive_us'):
            if r[key] is not None and duration(r[key]) is None:raise ValueError('invalid module duration')
        if r['cpu_exclusive_us'] is not None and (r['cpu_total_us'] is None or r['cpu_exclusive_us']>r['cpu_total_us']):
            raise ValueError('invalid module exclusive duration')
        if r['cuda_activity_us'] is not None:raise ValueError('unverified CUDA module duration')
        if type(r['annotation']) is not str or not r['annotation'] or r['annotation'] in annotations:raise ValueError('duplicate/invalid annotation')
        annotations.add(r['annotation'])
        if parent is None:
            counts[index]+=1
            if r['module_id']!=roots[0]['module_id']:raise ValueError('invocation root is not model root')
        else:
            if type(parent) is not int or not 0<=parent<i or rows[parent]['measurement_index']!=index:
                raise ValueError('invalid/cross-forward invocation parent')
            p=modules[rows[parent]['module_id']];m=modules[r['module_id']]
            if not any((alias.rpartition('.')[0] if '.' in alias else '')==pa and alias!=pa
                       for alias in m['aliases'] for pa in p['aliases']):raise ValueError('module is not a direct child')
    if counts!=[1]*forwards:raise ValueError('incomplete module forward ledger')


def module_timings(inventory,rows,events,forwards):
    validate_modules(inventory,rows,forwards);by_id=validate_events(events)
    scopes={}
    for e in events:
        if e['device_type']=='CPU':scopes.setdefault(e['name'],[]).append(e)
    result=[];selected=[]
    for r in rows:
        matches=scopes.get(r['annotation'],[])
        if len(matches)>1:raise ValueError('ambiguous module event')
        e=matches[0] if matches else None;selected.append(e)
        value=e['cpu_total_us'] if e else None
        if e and e.get('start_us') is not None and e.get('end_us') is not None and value is not None:
            if not math.isclose(value,e['end_us']-e['start_us'],rel_tol=1e-7,abs_tol=1e-6):
                raise ValueError('module duration disagrees with event interval')
        result.append(dict(r,cpu_total_us=value,cpu_exclusive_us=None,cuda_activity_us=None,
            cpu_event_id=e['event_id'] if e else None,cpu_timing_reason='missing_or_invalid_module_event' if value is None else None))
    for i,r in enumerate(result):
        e=selected[i];parent=r['parent_invocation_id']
        if e and parent is not None and selected[parent]:
            p=selected[parent];ancestors=set();pid=e['parent_id']
            while pid is not None:
                ancestors.add(pid);pid=by_id[('CPU',pid)]['parent_id']
            if p['event_id'] not in ancestors:raise ValueError('module/event parent mismatch')
        children=[j for j,v in enumerate(rows) if v['parent_invocation_id']==i]
        if r['cpu_total_us'] is None:continue
        if e.get('start_us') is None or e.get('end_us') is None:
            r['cpu_timing_reason']='missing_module_interval';continue
        if any(result[j]['cpu_total_us'] is None for j in children):
            r['cpu_timing_reason']='missing_or_invalid_child_duration';continue
        intervals=[];supported=True
        for j in children:
            child=selected[j]
            if (not e or not child or any(v is None for v in
                (e.get('start_us'),e.get('end_us'),child.get('start_us'),child.get('end_us')))
                or child.get('thread_id')!=e.get('thread_id')):
                supported=False;break
            if child['start_us']<e['start_us']-1e-6 or child['end_us']>e['end_us']+1e-6:
                raise ValueError('module child outside parent interval')
            intervals.append((child['start_us'],child['end_us']))
        intervals.sort()
        if not supported or any(right[0]<left[1]-1e-6 for left,right in zip(intervals,intervals[1:])):
            r['cpu_timing_reason']='overlapping_or_cross_thread_or_missing_child_intervals';continue
        exclusive=r['cpu_total_us']-sum(result[j]['cpu_total_us'] for j in children)
        r['cpu_exclusive_us']=duration(exclusive)
        if r['cpu_exclusive_us'] is None:r['cpu_timing_reason']='negative_exclusive_duration'
    return result


def cuda_event_timings(rows,evidence,device):
    """Validate a worker's serialized event ledger, not the original GPU clock.

    Inclusive stream elapsed intervals include idle/launch/instrumentation gaps.
    Kernel attribution and exclusive CUDA work cannot be inferred from them.
    """
    if type(evidence.get('version')) is not int or evidence['version']!=1:
        raise ValueError('invalid CUDA Event evidence version')
    requested=device.startswith('cuda:')
    if type(evidence['requested']) is not bool or evidence['requested']!=requested or evidence['device']!=device:
        raise ValueError('CUDA Event device/request mismatch')
    if evidence['clock']!='torch.cuda.Event.elapsed_time_ms' or evidence['scope']!='owned_eager_single_host_thread_and_stream; unmanaged_native_async_excluded':
        raise ValueError('invalid CUDA Event clock/scope')
    if evidence['validation']!='serialized_ledger_and_ms_to_us_only; CUDA_event_objects_not_serialized_or_replayed':
        raise ValueError('invalid CUDA Event validation claim')
    pairs=evidence['pairs'];reason=evidence['scope_reason']
    reasons={'multiple_or_changed_CUDA_streams','unverified_async_Python_thread',
        'unverified_existing_Python_threads','cross_thread_module_or_dispatch','unverified_custom_CUDA_dispatch'}
    if reason is not None and reason not in reasons:raise ValueError('invalid CUDA Event scope reason')
    if not requested:
        if (pairs or evidence['stream_id'] is not None or evidence['host_thread_id'] is not None
                or evidence['synchronized'] is not False or reason is not None):raise ValueError('CPU capture cannot claim CUDA Events')
    elif (not _id(evidence['stream_id']) or not _id(evidence['host_thread_id'])
            or evidence['synchronized'] is not True or len(pairs)!=len(rows)):
        raise ValueError('incomplete CUDA Event capture/stream ledger')
    result=[]
    for i,r in enumerate(rows):
        value=None;status='not_requested';why=None
        if requested:
            p=pairs[i]
            for key,expected in (('invocation_id',i),('measurement_index',r['measurement_index']),
                                 ('start_event_index',2*i),('end_event_index',2*i+1)):
                if type(p[key]) is not int or p[key]!=expected:raise ValueError('CUDA Event invocation/pair association mismatch')
            if p['device']!=device:raise ValueError('CUDA Event pair on wrong device')
            for key in ('start_stream_id','end_stream_id','start_thread_id','end_thread_id'):
                if p[key] is not None and not _id(p[key]):raise ValueError('invalid CUDA Event stream/thread ID')
            for key in ('start_recorded','end_recorded','start_complete','end_complete'):
                if type(p[key]) is not bool:raise ValueError('invalid CUDA Event completion flag')
            for prefix in ('start','end'):
                if p[prefix+'_complete'] and not p[prefix+'_recorded']:raise ValueError('unrecorded CUDA Event cannot be complete')
            elapsed=p['elapsed_ms']
            if elapsed is not None and (type(elapsed) not in (int,float) or not math.isfinite(elapsed)):
                raise ValueError('invalid serialized CUDA Event elapsed time')
            if p['error'] not in (None,'nonpositive_or_nonfinite_CUDA_event_elapsed',
                'CUDA_event_elapsed_failed','missing_or_incomplete_CUDA_event_pair'):
                raise ValueError('invalid CUDA Event error reason')
            if reason is not None:
                if elapsed is not None:raise ValueError('unsupported CUDA scope cannot claim elapsed time')
                status='unsupported';why=reason
            else:
                if (p['start_stream_id']!=evidence['stream_id'] or p['end_stream_id']!=evidence['stream_id']
                        or p['start_thread_id']!=evidence['host_thread_id'] or p['end_thread_id']!=evidence['host_thread_id']):
                    raise ValueError('CUDA Event stream/thread mismatch')
                complete=all(p[k] for k in ('start_recorded','end_recorded','start_complete','end_complete'))
                if complete and elapsed is not None and elapsed>0 and p['error'] is None:
                    value=elapsed*1000.
                    if not math.isfinite(value):raise ValueError('CUDA Event microsecond conversion overflow')
                    status='measured'
                else:
                    if elapsed is not None and elapsed>0:raise ValueError('unverified CUDA Event interval cannot be numeric')
                    status='invalid';why=p['error'] or 'missing_or_nonpositive_CUDA_event_elapsed'
        result.append(dict(r,cuda_elapsed_us=value,cuda_elapsed_status=status,cuda_elapsed_reason=why,
            cuda_exclusive_elapsed_us=None,cuda_exclusive_elapsed_reason='not_derived_from_inclusive_CUDA_Event_intervals' if requested else 'not_requested'))
    return result


def normalize_trace(trace,max_events):
    rows=[];kernels=[]
    for index,e in enumerate(trace.get('traceEvents',[])):
        if e.get('ph')!='X' or e.get('cat') not in ('cpu_op','user_annotation','cuda_runtime','cuda_driver','kernel'):continue
        if len(rows)>=max_events:raise ValueError('normalized trace event budget exceeded')
        args=e.get('args',{});start=duration(e.get('ts'));elapsed=duration(e.get('dur'))
        row=dict(trace_index=index,category=e['cat'],phase='X',name=e.get('name'),
            start_us=start,duration_us=elapsed,pid=e.get('pid'),tid=e.get('tid'),
            raw_correlation=args.get('correlation'),raw_external_id=args.get('External id'),
            raw_event_id=e.get('id'),device=args.get('device'),stream=args.get('stream'))
        rows.append(row)
        if e['cat']=='kernel':
            kernels.append(dict(name=e.get('name'),duration_us=elapsed,category='kernel',phase='X',
                status='measured' if elapsed is not None else 'invalid_duration',
                raw_correlation=row['raw_correlation'],raw_external_id=row['raw_external_id'],
                device=row['device'],stream=row['stream'],correlation_status='raw_trace_ids_only_not_verified',
                trace_index=index,start_us=start,timestamp_domain='same_capture_chrome_trace_microseconds',
                raw_event_id=row['raw_event_id']))
    return rows,kernels


def correlate(rows,invocations):
    """Kernel -> unique launch -> unique external CPU op -> module/root scope.

    Only CPU launch/op/annotation intervals are compared. A GPU kernel need not
    overlap its CPU module. Unknown/fused/graph launches are never distributed.
    """
    by_index={};ops={};launches={};annotations={};kernel_ids=set()
    for e in rows:
        if not _id(e['trace_index']) or e['trace_index'] in by_index:raise ValueError('duplicate/invalid trace event index')
        by_index[e['trace_index']]=e
        if e['phase']!='X' or type(e['name']) is not str or not e['name']:raise ValueError('invalid trace event type/name')
        for k in ('start_us','duration_us'):
            if e[k] is not None and duration(e[k]) is None:raise ValueError('invalid trace time')
        if e['category']=='kernel':
            identity=tuple(e[k] for k in ('raw_correlation','device','stream','start_us','duration_us','name'))
            if identity in kernel_ids:raise ValueError('duplicate kernel activity')
            kernel_ids.add(identity)
        if e['category']=='cpu_op' and _id(e['raw_external_id']):ops.setdefault(e['raw_external_id'],[]).append(e)
        if e['category'] in ('cuda_runtime','cuda_driver') and _id(e['raw_correlation']):launches.setdefault(e['raw_correlation'],[]).append(e)
        if e['category']=='user_annotation':annotations.setdefault(e['name'],[]).append(e)
    def contains(outer,inner):
        return (all(_id(v) for v in (outer['pid'],outer['tid'],inner['pid'],inner['tid']))
            and outer['pid']==inner['pid'] and outer['tid']==inner['tid']
            and all(v is not None for v in (outer['start_us'],outer['duration_us'],inner['start_us'],inner['duration_us']))
            and outer['start_us']<=inner['start_us']
            and inner['start_us']+inner['duration_us']<=outer['start_us']+outer['duration_us']+1e-6)
    matches=[]
    for k in rows:
        if k['category']!='kernel':continue
        r=dict(kernel_trace_index=k['trace_index'],operator_trace_index=None,operator_name=None,
               invocation_id=None,measurement_index=None,status='unsupported',reason='missing_or_ambiguous_launch_correlation')
        matches.append(r)
        if not all(_id(k[key]) for key in ('device','stream','raw_correlation','raw_external_id')) or k['start_us'] is None or k['duration_us'] is None:
            r['reason']='invalid_kernel_identity_or_time';continue
        launch=launches.get(k['raw_correlation'],[])
        if len(launch)!=1:continue
        launch=launch[0]
        if not launch['name'].startswith(('cudaLaunchKernel','cuLaunchKernel')):
            r['reason']='unsupported_launch_kind';continue
        candidates=ops.get(k['raw_external_id'],[])
        if len(candidates)!=1 or launch['raw_external_id']!=k['raw_external_id']:
            r['reason']='missing_ambiguous_or_mismatched_external_id';continue
        op=candidates[0]
        if not contains(op,launch) or launch['start_us']>k['start_us']:
            r['reason']='inconsistent_operator_launch_or_timestamp';continue
        scopes=[]
        for v in invocations:
            annotation=annotations.get(v['annotation'],[])
            if len(annotation)==1 and contains(annotation[0],op):scopes.append(v)
        roots=[v for v in scopes if v['parent_invocation_id'] is None]
        if len(roots)!=1:
            r['reason']='missing_or_ambiguous_forward_scope';continue
        # The matching scopes must form one ancestry chain, never siblings.
        ids={v['invocation_id'] for v in scopes};deepest=max(scopes,key=lambda v:v['invocation_id'])
        chain=set();current=deepest
        while True:
            chain.add(current['invocation_id']);parent=current['parent_invocation_id']
            if parent is None:break
            current=invocations[parent]
        if chain!=ids:
            r['reason']='ambiguous_or_incomplete_module_chain';continue
        r.update(operator_trace_index=op['trace_index'],operator_name=op['name'],
            invocation_id=deepest['invocation_id'],measurement_index=roots[0]['measurement_index'],status='mapped',reason=None)
    mapped=[m for m in matches if m['status']=='mapped'];groups={}
    for m in mapped:groups.setdefault(m['operator_name'],[]).append(m['kernel_trace_index'])
    operators=[dict(name=name,kernel_trace_indices=indices,
        observed_kernel_us=math.fsum(by_index[i]['duration_us'] for i in indices)) for name,indices in sorted(groups.items())]
    return dict(status='unsupported' if not mapped else 'complete' if len(mapped)==len(matches) else 'partial',
        reason='no_verified_operator_launch_kernel_chain' if not mapped else 'unattributed_kernels' if len(mapped)<len(matches) else None,
        mapped_kernel_count=len(mapped),unmapped_kernel_count=len(matches)-len(mapped),matches=matches,operators=operators,
        timestamp_domain='same_capture_chrome_trace_microseconds',
        accounting='each observed kernel assigned once; sums are device activity, not wall time or inference latency')
