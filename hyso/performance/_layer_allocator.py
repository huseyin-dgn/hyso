"""Worker-owned allocation lifetimes; allocation origin is not Tensor ownership.

No saved-tensor hooks, allocator peak resets or parent CUDA objects are used.
CUDA history is optional and private-API gated; CPU witnesses hold weak refs only.
"""
import hashlib
import json
import weakref


def _digest(blocks):
    return hashlib.sha256(json.dumps(sorted(blocks),separators=(',',':')).encode()).hexdigest()


class AllocationRecorder:
    def __init__(self,torch,device,budget):
        self.torch=torch;self.device=device;self.budget=budget;self.refs={};self.exits=[];self.points=[]
        self.history=False;self.initial=[];self.initial_requested=[];self.initial_position=0;self.reason=None
        if device.type=='cuda':
            if torch.cuda.get_allocator_backend()!='native':self.reason='CUDA_allocator_history_requires_native_backend'
            elif not all(hasattr(torch.cuda.memory,k) for k in ('_snapshot','_record_memory_history')):
                self.reason='CUDA_allocator_history_API_unavailable'
            else:
                torch.cuda.memory._record_memory_history(enabled='all',context=None,stacks='python',max_entries=budget)
                self.history=True
                point,blocks=self._point(None,'initial');self.initial=blocks;self.initial_position=point['position']

    def observe(self,value,depth=0):
        if depth>8:return
        if isinstance(value,self.torch.Tensor) and value.layout==self.torch.strided:
            storage=value.untyped_storage();key=int(storage._cdata)
            self.refs[key]=weakref.ref(storage)
            if len(self.refs)>self.budget:raise ValueError('storage witness budget exceeded')
        elif type(value) in (list,tuple):
            for item in value:self.observe(item,depth+1)
        elif type(value) is dict:
            for item in value.values():self.observe(item,depth+1)

    def _point(self,index,phase):
        snapshot=self.torch.cuda.memory._snapshot();trace=snapshot['device_traces'][self.device.index]
        if len(trace)>=self.budget-1:raise ValueError('CUDA allocator history truncated or evidence budget exhausted')
        blocks=[(int(b['address']),int(b['size'])) for s in snapshot['segments'] if s['device']==self.device.index
                for b in s['blocks'] if b['state']=='active_allocated']
        if phase=='initial':
            self.initial_requested=[(int(b['address']),int(b['requested_size'])) for s in snapshot['segments'] if s['device']==self.device.index
                for b in s['blocks'] if b['state']=='active_allocated']
        current=int(self.torch.cuda.memory_allocated(self.device))
        if sum(size for _,size in blocks)!=current:raise ValueError('CUDA snapshot/counter disagreement')
        point=dict(invocation_id=index,phase=phase,position=len(trace),allocated_bytes=current,active_blocks_sha256=_digest(blocks))
        self.points.append(point)
        return point,blocks

    def begin(self,index,args):
        self.observe(args)
        if self.history:self._point(index,'entry')

    def end(self,index,out,modules):
        self.observe(out)
        # Direct dictionaries/containers only: no arbitrary properties or user methods.
        for _,module in modules:self.observe(vars(module))
        witnesses=[]
        for key,reference in list(self.refs.items()):
            storage=reference()
            if storage is None:del self.refs[key];continue
            witnesses.append(dict(storage_id=key,address=int(storage.data_ptr()),bytes=int(storage.nbytes()),device=str(storage.device)))
        self.exits.append(dict(invocation_id=index,storages=witnesses))
        if self.history:self._point(index,'exit')

    def result(self):
        history=[]
        if self.history:
            self._point(None,'final')
            history=[{k:e[k] for k in ('action','addr','size','stream') if k in e}
                     for e in self.torch.cuda.memory._snapshot()['device_traces'][self.device.index]]
            if len(history)>=self.budget-1:raise ValueError('CUDA allocator history overflow')
        return dict(version=3,storage_exits=self.exits,cuda=dict(status='recorded' if self.history else 'unsupported',
            reason=self.reason or ('CPU_has_no_global_allocator_history_API' if self.device.type=='cpu' else None),
            initial_blocks=self.initial,initial_requested_sizes=self.initial_requested,
            initial_position=self.initial_position,points=self.points,history=history),
            collector='weak_storage_witnesses_and_owned_native_CUDA_allocator_history',collector_peak_reset_called=False)

    def close(self):
        if self.history:self.torch.cuda.memory._record_memory_history(enabled=None)
        self.refs.clear()


def _cuda_metrics(evidence,invocations,*,memory_events=None,device_index=0):
    if evidence['status']!='recorded':return dict(status='unsupported',reason=evidence['reason'],invocations=[])
    history=evidence['history'];initial=evidence['initial_position']
    for addr,size in evidence['initial_blocks']:
        if type(addr) is not int or addr<=0 or type(size) is not int or size<=0:raise ValueError('invalid allocator baseline block')
    live={addr:(size,-i-1) for i,(addr,size) in enumerate(evidence['initial_blocks'])}
    requested=dict(evidence.get('initial_requested_sizes',evidence['initial_blocks']))
    if len(live)!=len(evidence['initial_blocks']):raise ValueError('duplicate baseline CUDA allocation')
    states={};starts={};results={};stack=[];allocated=sum(s for s,_ in live.values())
    positions={};actual_sizes={};proof={}
    if memory_events is not None:
        queues={}
        for e in sorted(memory_events,key=lambda e:(e['start_us'],e['trace_index'])):
            if e['device_type']==1 and e['device_id']==device_index and e['bytes']>0:
                queues.setdefault(e['address'],[]).append(e)
        used={}
        for cursor,e in enumerate(history[initial:],initial):
            if e['action']!='alloc':continue
            addr=e['addr'];index=used.get(addr,0);candidates=queues.get(addr,[])
            if index>=len(candidates):raise ValueError('CUDA history allocation lacks actual profiler block-size evidence')
            event=candidates[index]
            if event['bytes']<e['size']:raise ValueError('CUDA requested bytes exceed actual block bytes')
            actual_sizes[cursor]=event['bytes'];proof[cursor]=event['trace_index'];used[addr]=index+1
        if any(used.get(addr,0)!=len(values) for addr,values in queues.items()):
            raise ValueError('CUDA profiler/history allocation-generation ledger mismatch')
    for point in evidence['points']:
        pos=point['position']
        if type(pos) is not int or not initial<=pos<=len(history):raise ValueError('invalid CUDA history checkpoint')
        if type(point['allocated_bytes']) is not int or point['allocated_bytes']<0:raise ValueError('invalid allocated-byte counter')
        positions.setdefault(pos,[]).append(point)
    for cursor in range(initial,len(history)+1):
        for point in positions.get(cursor,[]):
            if allocated!=point['allocated_bytes'] or _digest([(a,s) for a,(s,_) in live.items()])!=point['active_blocks_sha256']:
                raise ValueError('CUDA history does not reconcile independent snapshot/counter')
            i=point['invocation_id'];phase=point['phase']
            if phase=='entry':
                if i in starts or i!=len(starts) or invocations[i]['parent_invocation_id']!=(stack[-1] if stack else None):raise ValueError('invalid allocator invocation tree')
                starts[i]=(cursor,allocated);states[i]=allocated;stack.append(i)
            elif phase=='exit':
                if not stack or stack.pop()!=i:raise ValueError('invalid allocator checkpoint nesting')
                start,baseline=starts[i]
                retained=[dict(address=a,bytes=s,allocation_event_index=g) for a,(s,g) in live.items() if start<=g<cursor]
                results[i]=dict(invocation_id=i,status='measured',baseline_allocated_bytes=baseline,
                    end_allocated_bytes=allocated,peak_allocated_bytes=states[i],
                    retained_new_allocation_bytes=sum(b['bytes'] for b in retained),retained_allocations=retained,
                    retained_scope='allocation_origin_within_inclusive_invocation_and_still_active_allocated_at_exit; not_unique_Tensor_ownership')
            elif phase not in ('initial','final'):raise ValueError('invalid allocator checkpoint phase')
        if cursor==len(history):break
        event=history[cursor];action=event['action']
        if action in ('alloc','free_requested') and (type(event['addr']) is not int or event['addr']<=0 or type(event['size']) is not int or event['size']<=0):
            raise ValueError('invalid CUDA allocator event identity/size')
        if action=='alloc':
            addr=event['addr'];size=actual_sizes.get(cursor,event['size'])
            if type(size) is not int or size<=0 or addr in live:raise ValueError('invalid CUDA allocation lifetime')
            live[addr]=(size,cursor);requested[addr]=event['size'];allocated+=size
        elif action=='free_requested':
            old=live.pop(event['addr'],None)
            size=requested.pop(event['addr'],None)
            if old is None or size!=event['size']:raise ValueError('missing/repeated CUDA free or truncated history')
            allocated-=old[0]
        elif action not in ('free_completed','segment_alloc','segment_free','snapshot','oom'):
            raise ValueError('unsupported CUDA history action')
        for i in stack:states[i]=max(states[i],allocated)
    if stack or len(results)!=len(invocations):raise ValueError('incomplete CUDA allocator invocation ledger')
    result=dict(status='measured',scope='process_native_PyTorch_CUDA_allocated_bytes; includes_existing_baseline; not_reserved_or_physical_VRAM',
        method='ordered_allocator_history_reconciled_with_boundary_snapshots_and_memory_allocated',
        invocations=[results[i] for i in range(len(invocations))],physical_peak_bytes=None)
    if memory_events is not None:
        result.update(allocation_block_size_proof=[dict(history_index=i,profiler_trace_index=v) for i,v in sorted(proof.items())],
            block_size_method='profiler_actual_block_bytes_matched_by_address_and_allocation_generation; history_size_is_requested_bytes')
    return result


def derive_allocator(evidence,memory_events,trace,invocations,inventory,device):
    if evidence['version']==4:
        from ._cpu_layer_memory import extend_cpu_result
        base=derive_allocator(dict(evidence,version=3),memory_events,trace,invocations,inventory,device)
        return extend_cpu_result(base,evidence,memory_events,trace,invocations,inventory)
    if evidence['version'] not in (1,2,3) or evidence['collector_peak_reset_called'] is not False:raise ValueError('invalid allocator collector contract')
    exits={r['invocation_id']:r['storages'] for r in evidence['storage_exits']}
    if len(evidence['storage_exits'])!=len(invocations) or len(exits)!=len(invocations) or set(exits)!=set(range(len(invocations))):
        raise ValueError('incomplete/duplicate storage exit witnesses')
    from ._layer_memory import attribute_memory
    assignments=attribute_memory(memory_events,trace,invocations,inventory,device,analysis_version=2)['assignments']
    owners={r['trace_index']:r for r in assignments};cohorts=[];live={};unknown_frees=0
    previous={}
    for event in sorted(memory_events,key=lambda e:(e['start_us'],e['trace_index'])):
        if event['device_type']!=0 or event['device_id']!=-1:continue
        addr=event['address'];size=event['bytes'];timestamp=event['start_us']
        if type(addr) is not int or addr<=0:raise ValueError('invalid CPU allocation address')
        if previous.get(addr)==timestamp:raise ValueError('CPU address reused at unresolved identical timestamp')
        previous[addr]=timestamp
        if size>0:
            if addr in live:raise ValueError('duplicate live CPU allocation address; incomplete evidence')
            row=dict(address=addr,bytes=size,start_us=timestamp,end_us=None,allocation_trace_index=event['trace_index'],
                inclusive_invocation_ids=owners[event['trace_index']]['inclusive_invocation_ids'])
            cohorts.append(row);live[addr]=row
        else:
            old=live.pop(addr,None)
            if old is None:unknown_frees+=1;continue
            if old['bytes']!=-size:raise ValueError('CPU allocation/free size mismatch')
            old['end_us']=timestamp
    cpu=[]
    for invocation in invocations:
        i=invocation['invocation_id'];spans=[t for t in trace if t['category']=='user_annotation' and t['name']==invocation['annotation']]
        if len(spans)!=1:cpu.append(dict(invocation_id=i,status='unsupported',reason='missing_module_span'));continue
        end=spans[0]['start_us']+spans[0]['duration_us']
        created=[c for c in cohorts if i in c['inclusive_invocation_ids']]
        candidates=[c for c in created if c['end_us'] is None or c['end_us']>=end]
        witnessed={s['address'] for s in exits[i] if s['device']=='cpu' and s['bytes']>0}
        confirmed=[c for c in candidates if c['address'] in witnessed and (c['end_us'] is None or c['end_us']>end)]
        changes={}
        for c in created:
            changes.setdefault(c['start_us'],[]).append(c['bytes'])
            if c['end_us'] is not None and c['end_us']<end:changes.setdefault(c['end_us'],[]).append(-c['bytes'])
        balance=low=high=0
        for _,values in sorted(changes.items()):
            high=max(high,balance+sum(v for v in values if v>0));balance+=sum(values);low=max(low,balance)
        cpu.append(dict(invocation_id=i,status='measured' if len(confirmed)==len(candidates) else 'partial',
            retained_new_allocation_bytes=sum(c['bytes'] for c in confirmed),
            unconfirmed_live_allocation_bytes=sum(c['bytes'] for c in candidates if c not in confirmed),
            retained_allocation_trace_indices=[c['allocation_trace_index'] for c in confirmed],
            retained_scope='observed_allocations_created_inside_invocation_with_live_storage_witness_at_exit; allocation_origin_not_unique_ownership',
            new_allocation_live_peak_bytes=low if low==high else None,new_allocation_peak_bounds_bytes=[low,high],
            global_CPU_allocator_peak_bytes=None,physical_RSS_peak_bytes=None))
    cuda=_cuda_metrics(evidence['cuda'],invocations,memory_events=memory_events if evidence['version']>=2 else None,
        device_index=int(device.split(':')[1]) if device.startswith('cuda:') else 0)
    if device.startswith('cuda:') and any(r.get('cuda_elapsed_status')=='unsupported' for r in invocations):
        cuda=dict(status='unsupported',reason='unverified_multi_stream_or_thread_scope',invocations=[])
    modules=[]
    for m in inventory:
        ids=[r['invocation_id'] for r in invocations if r['module_id']==m['module_id']]
        values=[r for r in cuda['invocations'] if r['invocation_id'] in ids]
        modules.append(dict(module_id=m['module_id'],invocation_ids=ids,
            maximum_CUDA_allocator_peak_bytes=max((r['peak_allocated_bytes'] for r in values),default=None),
            retained_aggregation='per_invocation_exit_only; never_sum_across_shared_calls_or_nested_modules'))
        if evidence['version']>=3:
            cpu_values=[r for r in cpu if r['invocation_id'] in ids]
            # Keep the existing module-row CSV contract; expose retained values
            # as per-invocation arrays rather than an invalid sum of live sets.
            modules[-1].update(CPU_retained_new_allocation_bytes_at_exits=[r.get('retained_new_allocation_bytes') for r in cpu_values],
                CUDA_retained_new_allocation_bytes_at_exits=[r['retained_new_allocation_bytes'] for r in values])
    return dict(status='partial',cpu=dict(status='measured' if cpu and all(r['status']=='measured' for r in cpu) else 'partial',
        unknown_preexisting_frees=unknown_frees,invocations=cpu),cuda=cuda,modules=modules,
        limitations=['Weak storage witnesses do not retain user tensors; aliased views share one storage/address.',
            'Retained bytes describe newly created allocation cohorts alive at that invocation exit, not an exclusive module owner or eventual lifetime.',
            'CUDA values are native caching-allocator allocated bytes, not physical VRAM, reserved bytes or other applications.',
            'CPU live-cohort peaks exclude pre-existing/unobserved allocations; global CPU allocator and physical RSS layer peaks are unsupported.',
            'record_shapes is disabled in lifetime mode to avoid known profiler Tensor-reference retention; dispatch metadata still supplies analytic shapes.',
            'Snapshots, weak-reference scans and profiler allocation tracing perturb scheduling and CPU memory; overhead is included.',
            'History is bounded, validated against snapshots/counters, and disabled in the owned worker on cleanup; no allocator peak reset.'])
