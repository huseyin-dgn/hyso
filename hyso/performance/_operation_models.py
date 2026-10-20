"""Logical analytic models of observed ATen calls; never hardware counters."""
import math


CONVENTION = dict(id='aten_logical_v1',mac='one multiply-accumulate',fma_flops=2,
    bias='one addition per output element',comparisons='separate; not FLOPs',
    memory_layout_copy_cast='zero arithmetic FLOPs; memory cost not counted',
    nonlinear_normalization='unknown unless explicitly modeled',
    scope='observed dispatch calls; dense logical work, not hardware instructions')


def _tensor(value):
    return type(value) is dict and value.get('kind')=='tensor'


def _shape(value):
    return value['shape'] if _tensor(value) else None


def _size(value):
    return math.prod(_shape(value)) if _tensor(value) else None


def cost(call):
    """Return null costs for unknowns; distinguish structural zero from unknown."""
    name=call['operator'].split('.')[1] if call['operator'].startswith('aten.') else '<custom_namespace>'
    a=call['arguments'];out=call['output']
    if type(out) is list and out:out=out[0]
    n=_size(out) if _tensor(out) and out['dtype'] in ('torch.float16','torch.float32','torch.float64','torch.bfloat16') else None
    macs=flops=comparisons=None;policy=None
    structural={'view','reshape','flatten','squeeze','unsqueeze','transpose','permute','t','detach','alias',
        'clone','contiguous','copy_','_to_copy','to','expand','slice','select','split','split_with_sizes',
        'cat','empty','empty_like','empty_strided','new_empty','ones','ones_like','zeros','zeros_like','fill_'}
    if name in structural:
        macs=flops=comparisons=0;policy='memory_layout_copy_cast_excluded'
    elif n is not None and name in ('conv1d','conv2d','convolution','_convolution'):
        w=a.get('weight');x=a.get('input');transposed=a.get('transposed',False)
        if _tensor(w) and _tensor(x) and len(w['shape']) in (3,4) and not transposed:
            macs=n*w['shape'][1]*math.prod(w['shape'][2:])
            flops=2*macs+(n if _tensor(a.get('bias')) else 0);comparisons=0
            policy='dense_output_times_Cin_per_group_times_kernel_plus_bias'
    elif n is not None and name=='linear' and _tensor(a.get('weight')):
        macs=n*a['weight']['shape'][-1];flops=2*macs+(n if _tensor(a.get('bias')) else 0)
        comparisons=0;policy='linear_dense_plus_bias'
    elif n is not None and name in ('mm','matmul','bmm','addmm','baddbmm','dot'):
        x=a.get('mat1',a.get('batch1',a.get('self',a.get('input'))))
        if _tensor(x) and a.get('alpha',1)==1 and a.get('beta',1) in (0,1):
            macs=n*x['shape'][-1];flops=2*macs
            if name in ('addmm','baddbmm') and a.get('beta',1)!=0:flops+=n
            comparisons=0;policy='dense_matmul_plus_optional_beta1_add'
    elif n is not None and name.rstrip('_') in ('add','sub','mul','div'):
        macs=0;flops=n*(2 if name.rstrip('_') in ('add','sub') and a.get('alpha',1)!=1 else 1)
        comparisons=0;policy='broadcast_output_scalar_arithmetic'
    elif n is not None and name in ('relu','relu_','clamp_min'):
        macs=flops=0;comparisons=n;policy='comparison_only_activation'
    elif n is not None and name in ('adaptive_avg_pool1d','adaptive_avg_pool2d','_adaptive_avg_pool2d'):
        x=a.get('self',a.get('input'))
        if _tensor(x):
            spatial=len(out['shape'])-2;sums=[]
            for src,dst in zip(x['shape'][-spatial:],out['shape'][-spatial:]):
                sums.append(sum(((i+1)*src+dst-1)//dst-(i*src)//dst for i in range(dst)))
            macs=0;flops=math.prod(x['shape'][:2])*math.prod(sums);comparisons=0
            policy='adaptive_bin_sum_k_minus_one_additions_and_one_division'
    elif n is not None and name in ('avg_pool1d','avg_pool2d','max_pool1d','max_pool2d','max_pool2d_with_indices'):
        padding=a.get('padding',[]);ceil=a.get('ceil_mode',False);kernel=a.get('kernel_size',[])
        if not any(padding) and not ceil and kernel:
            k=math.prod(kernel);macs=0
            flops=n*k if name.startswith('avg') else 0
            comparisons=n*(k-1) if name.startswith('max') else 0;policy='full_nonpadded_fixed_pool_windows'
    elif name=='dropout' and not a.get('train',True):
        macs=flops=comparisons=0;policy='inference_dropout_identity'
    reason=None if flops is not None else 'operator_or_scalar_policy_not_modeled'
    return dict(call_index=call['call_index'],measurement_index=call['measurement_index'],
        operator=call['operator'],arguments=call['arguments'],output=call['output'],
        macs=macs,flops=flops,comparisons=comparisons,status='modeled' if flops is not None else 'unknown',
        policy=policy,reason=reason)


def summarize(calls,forwards,batch):
    rows=[cost(c) for c in calls];known=[r for r in rows if r['flops'] is not None]
    unknown=len(rows)-len(known);macs=sum(r['macs'] for r in known);flops=sum(r['flops'] for r in known)
    comparisons=sum(r['comparisons'] for r in known)
    m=dict(status='partial' if unknown else 'complete',reason='unmodeled_observed_calls' if unknown else None,
        total_macs_per_batch=None if unknown else macs/forwards,
        total_flops_per_batch=None if unknown else flops/forwards,
        total_flops_per_sample=None if unknown else flops/(forwards*batch),
        supported_subtotal_macs=macs,supported_subtotal_flops=flops,
        supported_flops_per_batch=flops/forwards,comparisons_per_batch=comparisons/forwards,
        coverage=dict(observed_calls=len(rows),modeled_calls=len(known),unknown_calls=unknown,
            modeled_call_fraction=len(known)/len(rows) if rows else 1.,
            basis='observed_ATen_call_count; not fraction of FLOPs or kernels'),
        unit='logical_operation_count',batch_cardinality=batch,
        aggregation='sum_over_profiled_forwards; per_batch_is_mean_of_observed_forwards',
        dynamic_control_flow='only_executed_paths_for_recipe_seed_input_and_inference_mode')
    m['per_forward']=[]
    for index in range(forwards):
        group=[r for r in rows if r['measurement_index']==index]
        complete=all(r['flops'] is not None for r in group)
        known_group=[r for r in group if r['flops'] is not None]
        m['per_forward'].append(dict(measurement_index=index,batch_cardinality=batch,
            total_macs=sum(r['macs'] for r in group) if complete else None,
            total_flops=sum(r['flops'] for r in group) if complete else None,
            supported_subtotal_flops=sum(r['flops'] for r in known_group),
            status='complete' if complete else 'partial'))
    return rows,m
