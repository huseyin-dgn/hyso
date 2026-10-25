"""CPU captured-allocator counters and observable references, never native heap.

Memory collection starts before model construction. Reference scans inspect only
plain module dictionaries/containers and call inputs/outputs, without gc referrer
walks or strong references retained across boundaries. CUDA uses the old adapter.
"""
import weakref
from bisect import bisect_left

from ._layer_allocator import AllocationRecorder


class CPUAllocationRecorder(AllocationRecorder):
    def __init__(self,torch,device,budget,modules,inventory):
        if device.type!='cpu':raise ValueError('CPU recorder requires CPU device')
        super().__init__(torch,device,budget)
        self.modules=modules;self.inventory=inventory;self.tensor_refs={};self.tensor_serial=0
        self.storage_serials={};self.storage_serial=0;self.cpu_points=[];self.input_edges={};self.witness_items=0

    def observe(self,value,depth=0):
        if depth>8:return
        if isinstance(value,self.torch.Tensor) and value.layout==self.torch.strided:
            super().observe(value)
            old=self.tensor_refs.get(id(value))
            if old is None or old[1]() is not value:
                self.tensor_serial+=1;self.tensor_refs[id(value)]=(self.tensor_serial,weakref.ref(value))
            if len(self.tensor_refs)>self.budget:raise ValueError('Tensor reference witness budget exceeded')
        elif type(value) in (list,tuple):
            for item in value:self.observe(item,depth+1)
        elif type(value) is dict:
            for item in value.values():self.observe(item,depth+1)

    def _edges(self,value,holder,path='',depth=0,ancestors=frozenset()):
        if depth>8:return []
        if isinstance(value,self.torch.Tensor) and value.layout==self.torch.strided:
            self.observe(value)
            return [dict(tensor_uid=self.tensor_refs[id(value)][0],holder=holder,path=path)]
        if type(value) not in (dict,list,tuple) or id(value) in ancestors:return []
        items=value.items() if type(value) is dict else enumerate(value)
        result=[]
        for key,item in items:
            label=str(key) if type(key) in (str,int) else '<opaque-key>'
            result+=self._edges(item,holder,path+'.'+label,depth+1,ancestors|{id(value)})
            if len(result)>self.budget:raise ValueError('ownership edge budget exceeded')
        return result

    def _boundary(self,index,phase,value):
        edges=self._edges(value,dict(kind='call_input' if phase=='entry' else 'call_output',invocation_id=index))
        if phase=='entry':self.input_edges[index]=edges.copy()
        else:edges+=self.input_edges.pop(index)
        for position,(_,module) in enumerate(self.modules):
            item=self.inventory[position]
            edges+=self._edges(vars(module),dict(kind='module_attribute',module_id=item['module_id'],
                module_aliases=item['aliases']))
        tensors=[];storages={};point_storage_ids={}
        for key,(uid,reference) in list(self.tensor_refs.items()):
            tensor=reference()
            if tensor is None:del self.tensor_refs[key];continue
            if tensor.device.type!='cpu':continue
            storage=tensor.untyped_storage();sid=int(storage._cdata)
            old=self.storage_serials.get(sid)
            if sid not in point_storage_ids and (old is None or old[1]() is not storage):
                self.storage_serial+=1;self.storage_serials[sid]=(self.storage_serial,weakref.ref(storage))
            storage_uid=point_storage_ids.setdefault(sid,self.storage_serials[sid][0])
            storages[storage_uid]=dict(storage_uid=storage_uid,storage_id=sid,address=int(storage.data_ptr()),bytes=int(storage.nbytes()),
                bytes_scope='untyped_storage_nbytes; not_native_heap_block_or_physical_residency')
            tensors.append(dict(tensor_uid=uid,storage_uid=storage_uid,shape=list(tensor.shape),
                stride=list(tensor.stride()),storage_offset=int(tensor.storage_offset())))
        live={t['tensor_uid'] for t in tensors};edges=[e for e in edges if e['tensor_uid'] in live]
        self.witness_items+=len(storages)+len(tensors)+len(edges)
        if self.witness_items>self.budget:raise ValueError('CPU boundary witness evidence budget exceeded')
        from .memory import _observe,_BackendUnavailable
        try:process=dict(status='measured',observation=_observe(),reason=None)
        except _BackendUnavailable as error:process=dict(status='unsupported',observation=None,reason=str(error))
        except ValueError as error:process=dict(status='measurement_error',observation=None,reason=str(error))
        self.cpu_points.append(dict(invocation_id=index,phase=phase,storages=list(storages.values()),
            tensors=tensors,references=edges,process_memory=process))

    def begin(self,index,args):
        super().begin(index,args);self._boundary(index,'entry',args)

    def end(self,index,out,modules):
        super().end(index,out,modules);self._boundary(index,'exit',out)

    def result(self):
        result=super().result();result['version']=4
        result['cpu_context']=dict(version=1,starts_before_model=True,points=self.cpu_points,
            reference_scope='plain_module_containers_and_call_boundary_references; no_global_or_native_reference_census',
            process_sampling='entry_and_exit_only; short_transients_may_be_missed')
        return result

    def close(self):
        self.tensor_refs.clear();self.storage_serials.clear();self.input_edges.clear();super().close()


def measured_subtree_events(events,invocations):
    """Full CPU trace seeds memory; operator rows retain measured-forward scope."""
    roots={r['annotation'] for r in invocations if r['parent_invocation_id'] is None}
    by_id={e['event_id']:e for e in events if e['device_type']=='CPU'}
    keep=set()
    for event in by_id.values():
        current=event;seen=set()
        while current is not None:
            if current['event_id'] in seen:raise ValueError('cyclic native CPU event tree')
            seen.add(current['event_id'])
            if current['name'] in roots:keep.add(event['event_id']);break
            current=by_id.get(current['parent_id'])
    return [dict(e,parent_id=e['parent_id'] if e['parent_id'] in keep else None)
            for e in events if e['device_type']=='CPU' and e['event_id'] in keep]


def _reference_graph(point,live,modules):
    storages={s['storage_uid']:s for s in point['storages']};tensors={t['tensor_uid']:t for t in point['tensors']}
    if len(storages)!=len(point['storages']) or len(tensors)!=len(point['tensors']):raise ValueError('duplicate storage/Tensor witness')
    for s in storages.values():
        if any(type(s[k]) is not int or s[k]<0 for k in ('storage_uid','storage_id','address','bytes')):raise ValueError('invalid storage identity/size')
    for t in tensors.values():
        if t['storage_uid'] not in storages:raise ValueError('Tensor refers to missing storage witness')
    for edge in point['references']:
        if edge['tensor_uid'] not in tensors:raise ValueError('reference refers to dead/unknown Tensor')
        holder=edge['holder'];kind=holder['kind']
        if kind=='module_attribute':
            if holder['module_id'] not in modules or holder['module_aliases']!=modules[holder['module_id']]['aliases']:
                raise ValueError('reference holder is not a verified module identity')
        elif kind in ('call_input','call_output'):
            if holder['invocation_id']!=point['invocation_id']:raise ValueError('reference from unrelated invocation')
        else:raise ValueError('unsupported reference holder kind')
    result=[]
    for uid,storage in storages.items():
        aliases=[t for t in tensors.values() if t['storage_uid']==uid]
        ids={t['tensor_uid'] for t in aliases};refs=[e for e in point['references'] if e['tensor_uid'] in ids]
        holders=sorted({e['holder']['module_id'] for e in refs if e['holder']['kind']=='module_attribute'})
        allocation=live.get(storage['address'])
        result.append(dict(storage,allocation_generation=allocation['allocation_trace_index'] if allocation else None,
            allocation_base_address=allocation['address'] if allocation else None,
            storage_allocation_offset_bytes=storage['address']-allocation['address'] if allocation else None,
            allocation_origin_invocation_id=allocation['origin'] if allocation else None,
            allocation_origin_scope='measured_invocation' if allocation and allocation['origin'] is not None else
                'capture_outside_measured_invocation' if allocation else 'outside_captured_allocation_ledger',
            tensor_aliases=aliases,observed_references=refs,observed_module_holders=holders,
            ownership='shared_observed' if len(holders)>1 else 'single_model_holder_observed' if holders else 'no_model_holder_observed',
            exclusive_owner=None,complete_reference_graph=False))
    groups={}
    for s in result:
        key=('allocation',s['allocation_generation']) if s['allocation_generation'] is not None else ('storage',s['storage_uid'])
        groups.setdefault(key,[]).append(s)
    for values in groups.values():
        holders=sorted({m for s in values for m in s['observed_module_holders']})
        for s in values:
            s.update(allocation_alias_storage_uids=[v['storage_uid'] for v in values],
                allocation_module_holders=holders,
                allocation_ownership='shared_observed' if len(holders)>1 else 'single_model_holder_observed' if holders else 'no_model_holder_observed')
    return result


def _witnessed_capacity(storages):
    """Union of live virtual storage spans, not a sum of aliased capacities."""
    ranges=sorted((s['address'],s['address']+s['bytes']) for s in storages if s['bytes'])
    total=0;start=end=None
    for lo,hi in ranges:
        if start is None:start,end=lo,hi
        elif lo<=end:end=max(end,hi)
        else:total+=end-start;start,end=lo,hi
    return total+(end-start if start is not None else 0)


def extend_cpu_result(base,evidence,memory_events,trace,invocations,inventory):
    context=evidence['cpu_context']
    if context['version']!=1 or context['starts_before_model'] is not True:raise ValueError('CPU baseline needs full setup capture')
    events=[e for e in memory_events if e['device_type']==0 and e['device_id']==-1]
    events=sorted(events,key=lambda e:(e['start_us'],e['trace_index']))
    if any(type(e['total_allocated_bytes']) is not int for e in events):raise ValueError('invalid CPU allocator counter type')
    balance=events[0]['total_allocated_bytes']-events[0]['bytes'] if events else 0
    if balance<0:raise ValueError('invalid initial CPU allocator counter')
    initial=balance;live={};lifetimes={};unknown_frees=0
    from ._layer_memory import attribute_memory
    owner={r['trace_index']:r['invocation_id'] for r in attribute_memory(memory_events,trace,invocations,inventory,'cpu',analysis_version=2)['assignments']}
    for event in events:
        size=event['bytes'];addr=event['address'];total=event['total_allocated_bytes']
        if type(total) is not int or total<0 or total!=balance+size:raise ValueError('CPU event loss/order/counter disagreement')
        if size>0:
            if addr in live:raise ValueError('live CPU allocation address reused without free')
            value=dict(address=addr,bytes=size,allocation_trace_index=event['trace_index'],origin=owner.get(event['trace_index']),start_us=event['start_us'],end_us=None)
            live[addr]=value;lifetimes.setdefault(addr,[]).append(value)
        else:
            previous=live.pop(addr,None)
            if previous is None:unknown_frees+=1
            elif previous['bytes']!=-size:raise ValueError('CPU allocation generation/free size disagreement')
            if previous is not None:previous['end_us']=event['start_us']
        balance=total
    points={(p['invocation_id'],p['phase']):p for p in context['points']}
    if any(type(p['invocation_id']) is not int for p in context['points']):raise ValueError('invalid CPU boundary invocation ID')
    if len(points)!=len(context['points']) or set(points)!={(i,phase) for i in range(len(invocations)) for phase in ('entry','exit')}:
        raise ValueError('incomplete/duplicate CPU boundary ledger')
    timestamps=[e['start_us'] for e in events]
    def state_at(timestamp,point):
        index=bisect_left(timestamps,timestamp)-1;known={}
        for s in point['storages']:
            candidates=[c for c in lifetimes.get(s['address'],[]) if c['start_us']<timestamp and (c['end_us'] is None or c['end_us']>timestamp)]
            if not candidates and s['bytes']:
                # A NumPy/frombuffer view can expose a distinct StorageImpl at
                # an interior pointer into a captured C10 allocation.
                candidates=[c for values in lifetimes.values() for c in values if c['start_us']<timestamp and
                    (c['end_us'] is None or c['end_us']>timestamp) and
                    c['address']<=s['address'] and s['address']+s['bytes']<=c['address']+c['bytes']]
            if len(candidates)>1:raise ValueError('ambiguous overlapping live CPU allocation ranges')
            if len(candidates)==1:known[s['address']]=candidates[0]
        return events[index]['total_allocated_bytes'] if index>=0 else initial,known
    rows=[]
    for r in invocations:
        i=r['invocation_id'];spans=[e for e in trace if e['category']=='user_annotation' and e['name']==r['annotation']]
        if len(spans)!=1:raise ValueError('CPU peak requires one verified module span')
        start=spans[0]['start_us'];end=start+spans[0]['duration_us']
        entry=points[i,'entry'];exit=points[i,'exit']
        baseline,entry_live=state_at(start,entry);after,exit_live=state_at(end,exit)
        inside=[e for e in events if start<e['start_us']<end]
        border=[e for e in events if e['start_us'] in (start,end)]
        peak=max([baseline]+[e['total_allocated_bytes'] for e in inside])
        modules={m['module_id']:m for m in inventory}
        graphs=dict(entry=_reference_graph(entry,entry_live,modules),exit=_reference_graph(exit,exit_live,modules))
        untracked=[s for s in graphs['entry'] if s['bytes'] and s['allocation_generation'] is None]
        observations=[p['process_memory']['observation'] for p in (entry,exit) if p['process_memory']['status']=='measured']
        rows.append(dict(invocation_id=i,status='measured' if not border else 'partial',
            baseline_captured_allocator_bytes=baseline if not border else None,
            end_captured_allocator_bytes=after if not border else None,peak_captured_allocator_bytes=peak if not border else None,
            observed_counter_max_bytes=peak,boundary_timestamp_ambiguous=bool(border),
            untracked_entry_storage_capacity_bytes=_witnessed_capacity(untracked),
            untracked_entry_storage_uids=[s['storage_uid'] for s in untracked],
            untracked_storage_scope='reference-witnessed storage capacity outside captured allocator ledger; never added to allocator peak',
            reference_graph=graphs,process_memory=dict(samples=[p['process_memory'] for p in (entry,exit)],
                sampled_max_rss_bytes=max((o['rss_bytes'] for o in observations),default=None),
                observed_lifetime_hwm_bytes=max((o['lifetime_hwm_bytes'] for o in observations),default=None),
                physical_layer_peak_bytes=None,sampling='entry_exit_only; short_peaks_may_be_missed; lifetime_hwm_includes_pre_history')))
    summaries=[]
    for module in inventory:
        calls=[row for row in rows if invocations[row['invocation_id']]['module_id']==module['module_id']]
        summaries.append(dict(module_id=module['module_id'],invocation_ids=[c['invocation_id'] for c in calls],
            peak_captured_allocator_bytes=max(c['peak_captured_allocator_bytes'] for c in calls)
                if calls and all(c['peak_captured_allocator_bytes'] is not None for c in calls) else None,
            exit_reference_graphs=[c['reference_graph']['exit'] for c in calls],physical_layer_peak_bytes=None))
    base['cpu_peak_ownership']=dict(status='measured' if all(r['status']=='measured' for r in rows) else 'partial',
        scope='C10_default_CPU_allocator_allocations_reported_during_full_capture; includes_model_setup_and_instrumentation; not_global_heap_or_physical_RAM',
        initial_unattributed_counter_bytes=initial,unknown_baseline_frees=unknown_frees,
        event_count=len(events),event_counter_reconciliation='every_signed_delta_matches_reported_Total_Allocated',
        invocations=rows,modules=summaries,
        limitations=['Profiler begins before model construction; allocations predating capture or bypassing C10 are excluded.',
            'Peak uses all captured allocation/free events, not periodic RSS samples. Counter or generation loss rejects the result.',
            'Reference paths are observable holders only; one observed holder never proves global exclusive ownership.',
            'Storage UID, allocation generation, origin invocation and reference holder are distinct domains.',
            'Native malloc/mmap/third-party heaps are outside this allocator counter. RSS includes them and observer overhead.',
            'Boundary RSS sampling may miss short peaks; process lifetime high-water is never called a layer peak.'])
    for m in base['modules']:m['CPU_peak_ownership']=next(s for s in summaries if s['module_id']==m['module_id'])
    return base
