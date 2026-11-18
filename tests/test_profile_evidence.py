"""Synthetic adapter unit tests; these are never real CUDA acceptance evidence."""
import copy

import pytest

from hyso.performance._profile_evidence import (validate_events,validate_modules,module_timings,
    normalize_trace,correlate)
from hyso.performance.profiling import _layer_metrics,_cuda_metrics


def inventory():
    return [dict(module_id='m0',path='',aliases=['']),dict(module_id='m1',path='branch',aliases=['branch']),
            dict(module_id='m2',path='branch.leaf',aliases=['branch.leaf'])]


def invocation(i,module,parent):
    return dict(invocation_id=i,module_id=module,parent_invocation_id=parent,measurement_index=0,
                annotation='module.'+str(i),cpu_total_us=None,cpu_exclusive_us=None,cuda_activity_us=None)


def event(i,start,end,parent=None,thread=1):
    return dict(event_id=i+1,name='module.'+str(i),device_type='CPU',scope=7,parent_id=parent,
        cpu_total_us=end-start,self_cpu_us=0,timing_status='measured',start_us=start,end_us=end,thread_id=thread)


def nested():
    rows=[invocation(0,'m0',None),invocation(1,'m1',0),invocation(2,'m2',1)]
    events=[event(0,0,100),event(1,10,70,1),event(2,20,40,2)]
    return rows,events


def test_uncalled_registered_container_does_not_invent_an_invocation():
    rows=[invocation(0,'m0',None),invocation(1,'m2',0)]
    values=module_timings(inventory(),rows,[event(0,0,100),event(1,10,70,1)],1)
    assert len(values)==2 and [v['cpu_exclusive_us'] for v in values]==[40,60]
    assert all(v['module_id']!='m1' for v in values)


def test_registered_sibling_is_not_a_descendant_of_the_call_parent():
    modules=inventory()+[dict(module_id='other',path='branch_other',aliases=['branch_other'])]
    rows=[invocation(0,'m0',None),invocation(1,'m1',0),invocation(2,'other',1)]
    with pytest.raises(ValueError,match='descendant'):validate_modules(modules,rows,1)


def test_nested_same_clock_exclusive_and_unexecuted_inventory():
    rows,events=nested();values=module_timings(inventory(),rows,events,1)
    assert [v['cpu_exclusive_us'] for v in values]==[40,40,20]
    assert sum(v['cpu_exclusive_us'] for v in values)==values[0]['cpu_total_us']
    extra=inventory()+[dict(module_id='unused',path='unused',aliases=['unused'])]
    evidence=dict(cuda_attribution=correlate([],rows),overhead={})
    m=_layer_metrics(extra,values,evidence=evidence)
    unused=next(r for r in m['modules'] if r['module_id']=='unused')
    assert unused['status']=='not_executed' and unused['cpu_total_us'] is None
    assert m['layer_memory']['allocation_bytes'] is None and m['layer_memory']['peak_bytes'] is None


@pytest.mark.parametrize('mutation',[
    lambda e:e.append(copy.deepcopy(e[0])),
    lambda e:e[1].update(parent_id=999),
    lambda e:e[0].update(parent_id=2),
    lambda e:e[1].update(event_id=True),
    lambda e:e[1].update(cpu_total_us=-1),
    lambda e:e[1].update(self_cpu_us=61),
    lambda e:e[1].update(cpu_total_us=float('inf')),
    lambda e:e[1].update(end_us=0),
    lambda e:e[1].update(timing_status='invalid_or_missing'),
])
def test_invalid_native_identity_parent_and_duration_rejected(mutation):
    _,events=nested();mutation(events)
    with pytest.raises(ValueError):validate_events(events)


@pytest.mark.parametrize('mutation',[
    lambda r:r[1].update(parent_invocation_id=2),
    lambda r:r[1].update(measurement_index=1),
    lambda r:r[1].update(module_id='m2'),
    lambda r:r[1].update(annotation='module.0'),
    lambda r:r[1].update(cpu_exclusive_us=-1),
    lambda r:r[1].update(cuda_activity_us=0),
])
def test_bad_module_hierarchy_and_unverified_cuda_rejected(mutation):
    rows,_=nested();mutation(rows)
    with pytest.raises(ValueError):validate_modules(inventory(),rows,1)


def test_missing_events_invalid_duration_and_parent_mismatch():
    rows,events=nested()
    values=module_timings(inventory(),rows,events[:1],1)
    assert values[0]['cpu_total_us']==100 and values[0]['cpu_exclusive_us'] is None
    assert all(v['cpu_total_us'] is None for v in values[1:])
    evidence=dict(cuda_attribution=correlate([],rows),overhead={})
    assert _layer_metrics(inventory(),values,evidence=evidence)['status']=='partial'
    events[2].update(cpu_total_us=None,timing_status='invalid_or_missing')
    values=module_timings(inventory(),rows,events,1)
    assert values[1]['cpu_exclusive_us'] is None and values[2]['cpu_total_us'] is None
    events[2].update(parent_id=1)
    with pytest.raises(ValueError,match='parent mismatch'):module_timings(inventory(),rows,events,1)


def test_overlapping_or_other_thread_children_are_not_subtracted():
    modules=inventory();modules[2].update(path='other',aliases=['other'])
    rows=[invocation(0,'m0',None),invocation(1,'m1',0),invocation(2,'m2',0)]
    events=[event(0,0,100),event(1,10,60,1),event(2,50,90,1)]
    values=module_timings(modules,rows,events,1)
    assert values[0]['cpu_exclusive_us'] is None and 'overlapping' in values[0]['cpu_timing_reason']
    events[2]=event(2,60,90,1,thread=2)
    assert module_timings(modules,rows,events,1)[0]['cpu_exclusive_us'] is None
    events[2]=event(2,90,110,1)
    with pytest.raises(ValueError,match='outside parent'):module_timings(modules,rows,events,1)


def test_negative_exclusive_is_null_not_clamped_zero():
    modules=inventory();modules[2].update(path='other',aliases=['other'])
    rows=[invocation(0,'m0',None),invocation(1,'m1',0),invocation(2,'m2',0)]
    events=[event(0,0,.3),event(1,0,.1,1),event(2,.1,.3,1)]
    events[2]['cpu_total_us']=.2
    values=module_timings(modules,rows,events,1)
    assert values[0]['cpu_exclusive_us'] is None and values[0]['cpu_timing_reason']=='negative_exclusive_duration'


def trace_fixture():
    def e(cat,name,ts,dur,**args):
        return dict(cat=cat,ph='X',name=name,ts=ts,dur=dur,pid=10,tid=20,args=args)
    trace=dict(traceEvents=[e('user_annotation','module.0',0,100),
        e('user_annotation','module.1',5,50),
        e('cpu_op','aten::mm',10,20,**{'External id':7}),
        e('cuda_runtime','cudaLaunchKernel',15,5,correlation=9,**{'External id':7}),
        e('kernel','test_unit_kernel',150,8,device=0,stream=7,correlation=9,**{'External id':7})])
    rows=[invocation(0,'m0',None),invocation(1,'m1',0)]
    return trace,rows


def test_unique_chain_gpu_exec_after_cpu_module_is_allowed_and_counted_once():
    trace,invocations=trace_fixture();rows,kernels=normalize_trace(trace,100)
    evidence=correlate(rows,invocations)
    assert evidence['status']=='complete' and evidence['mapped_kernel_count']==1
    assert evidence['matches'][0]['operator_trace_index']==2
    assert evidence['matches'][0]['invocation_id']==1 and evidence['matches'][0]['measurement_index']==0
    assert sum(r['observed_kernel_us'] for r in evidence['operators'])==8
    assert _cuda_metrics(kernels,True)['total_kernel_us']==8
    # A second CPU operator with a different external ID receives no duration.
    trace['traceEvents'].append(dict(trace['traceEvents'][2],args={'External id':8}))
    rows,_=normalize_trace(trace,100)
    assert len(correlate(rows,invocations)['operators'])==1


@pytest.mark.parametrize('mutation',[
    lambda e:e[3]['args'].update(correlation=99),
    lambda e:e[3]['args'].update(**{'External id':99}),
    lambda e:e.append(copy.deepcopy(e[3])),
    lambda e:e.append(copy.deepcopy(e[2])),
    lambda e:e[3].update(tid=999),
    lambda e:e[3].update(ts=50),
    lambda e:e[3].update(name='cudaGraphLaunch'),
    lambda e:e[4].update(ts=1),
    lambda e:e[4]['args'].update(stream=None),
    lambda e:e[4].update(dur=None),
    lambda e:e[0].update(ts=200),
    lambda e:e.append(copy.deepcopy(e[0])),
])
def test_mismatch_ambiguity_invalid_identity_or_outside_forward_is_unknown(mutation):
    trace,invocations=trace_fixture();mutation(trace['traceEvents']);rows,kernels=normalize_trace(trace,100)
    result=correlate(rows,invocations)
    assert result['mapped_kernel_count']==0 and result['matches'][0]['operator_trace_index'] is None
    assert result['matches'][0]['reason'] and result['operators']==[]
    if kernels[0]['duration_us'] is None or kernels[0]['stream'] is None:
        assert _cuda_metrics(kernels,True)['total_kernel_us'] is None


def test_duplicate_kernel_index_and_budget_are_rejected():
    trace,invocations=trace_fixture();rows,_=normalize_trace(trace,100)
    with pytest.raises(ValueError):correlate(rows+[rows[-1]],invocations)
    duplicate=copy.deepcopy(rows[-1]);duplicate['trace_index']=100
    with pytest.raises(ValueError,match='duplicate kernel'):correlate(rows+[duplicate],invocations)
    with pytest.raises(ValueError,match='budget'):normalize_trace(trace,2)


def test_invalid_duration_is_partial_and_legacy_cuda_reason_preserved():
    old=dict(name='unit_kernel',duration_us=None,category='kernel',phase='X')
    assert _cuda_metrics([old],True)['reason']=='invalid_kernel_duration'
    trace,_=trace_fixture();trace['traceEvents'][-1]['ts']=None
    _,kernels=normalize_trace(trace,100)
    assert _cuda_metrics(kernels,True)['status']=='partial'
    assert _cuda_metrics(kernels,True)['total_kernel_us'] is None
