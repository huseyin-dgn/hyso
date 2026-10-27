"""Logical analytic models of observed ATen calls; never hardware counters."""
import math


CONVENTION = dict(id='aten_logical_v1',mac='one multiply-accumulate',fma_flops=2,
    bias='one addition per output element',comparisons='separate; not FLOPs',
    memory_layout_copy_cast='zero arithmetic FLOPs; memory cost not counted',
    nonlinear_normalization='unknown unless explicitly modeled',
    scope='observed dispatch calls; dense logical work, not hardware instructions')

CONVENTION_V3 = dict(CONVENTION, id='aten_logical_v3',
    flops='add/subtract/multiply/divide only; FMA=2; comparisons, roots and transcendentals separate',
    nonlinear_normalization='unfused reference algebra; named special operations have no FLOP weight',
    transposed_convolution='valid input-kernel-output contributions; clipped scatter, FMA=2',
    scalar_policy='dense reference evaluates alpha/beta scaling; no value-based runtime shortcut inference')


def _tensor(value):
    return type(value) is dict and value.get('kind')=='tensor'


def _shape(value):
    return value['shape'] if _tensor(value) else None


def _size(value):
    return math.prod(_shape(value)) if _tensor(value) else None


def cost(call,*,model_version=2):
    """Return null costs for unknowns; distinguish structural zero from unknown."""
    if type(model_version) is not int:raise ValueError('unsupported operation model version')
    if model_version==3:
        return _expanded_cost(call)
    if model_version==2:
        return _current_cost(call)
    if model_version!=1:raise ValueError('unsupported operation model version')
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


_FLOATS=('torch.float16','torch.float32','torch.float64','torch.bfloat16')


def _metadata(value):
    if _tensor(value):
        shape=value.get('shape')
        return (type(shape) is list and all(type(v) is int and v>=0 for v in shape)
                and type(value.get('dtype')) is str and type(value.get('device')) is str)
    if type(value) is dict:return all(_metadata(v) for v in value.values())
    if type(value) is list:return all(_metadata(v) for v in value)
    return value is None or type(value) in (str,bool,int) or _scalar(value)


def _scalar(value):
    return type(value) is int or type(value) is float and math.isfinite(value)


def _broadcast(left,right):
    result=[]
    for i in range(1,max(len(left),len(right))+1):
        x=left[-i] if i<=len(left) else 1;y=right[-i] if i<=len(right) else 1
        if x!=y and x!=1 and y!=1:return None
        result.append(y if x==1 else x)
    return list(reversed(result))


def _current_cost(call):
    # v1 remains an immutable reader for previously exported results. Validate
    # operands before reusing its formulas; never infer an absent tensor/scalar.
    row=dict(call,macs=None,flops=None,comparisons=None,status='unknown',policy=None,
             reason='operator_or_scalar_policy_not_modeled')
    def known(macs,flops,comparisons,policy):
        return dict(row,macs=macs,flops=flops,comparisons=comparisons,status='modeled',policy=policy,reason=None)
    if type(call['arguments']) is not dict or not _metadata(call['arguments']) or not _metadata(call['output']):
        return dict(row,reason='invalid_operand_metadata')
    a=call['arguments'];out=call['output']
    if type(out) is list and out:out=out[0]
    parts=call['operator'].split('.')
    name=parts[1] if len(parts)>=2 and parts[0]=='aten' else '<custom_namespace>'
    # Structural operations explicitly exclude arithmetic regardless of dtype.
    if name in {'view','reshape','flatten','squeeze','unsqueeze','transpose','permute','t','detach','alias',
        'clone','contiguous','copy_','_to_copy','to','expand','slice','select','split','split_with_sizes',
        'cat','empty','empty_like','empty_strided','new_empty','ones','ones_like','zeros','zeros_like','fill_'}:
        return known(0,0,0,'memory_layout_copy_cast_excluded')
    if not _tensor(out) or out['dtype'] not in _FLOATS:return row
    n=_size(out);shape=out['shape']
    def tensor(v):
        return _tensor(v) and (v['dtype'],v['device'])==(out['dtype'],out['device'])
    def bias(v,channels):
        return v is None or tensor(v) and v['shape']==[channels]
    x=a.get('input',a.get('self'))
    if name in ('conv1d','conv2d','convolution','_convolution'):
        w=a.get('weight');g=a.get('groups');rank=len(shape)
        if not (tensor(x) and tensor(w) and rank in (3,4) and len(x['shape'])==rank==len(w['shape'])
                and type(g) is int and g>0 and 'bias' in a and bias(a['bias'],w['shape'][0])
                and x['shape'][1]==g*w['shape'][1] and w['shape'][0]%g==0
                and shape[:2]==[x['shape'][0],w['shape'][0]]):return row
        if name in ('convolution','_convolution') and a.get('transposed') is not False:return row
        stride=a.get('stride');padding=a.get('padding');dilation=a.get('dilation');d=rank-2
        if not all(type(v) is list and len(v)==d and all(type(t) is int and t>=minimum for t in v)
                   for v,minimum in ((stride,1),(padding,0),(dilation,1))):return row
        expected=[(src+2*p-dil*(k-1)-1)//s+1 for src,k,s,p,dil in
                  zip(x['shape'][2:],w['shape'][2:],stride,padding,dilation)]
        if any(k<=0 for k in w['shape'][2:]) or shape[2:]!=expected:return row
        macs=n*w['shape'][1]*math.prod(w['shape'][2:])
        return known(macs,2*macs+(n if tensor(a['bias']) else 0),0,
                     'dense_output_times_Cin_per_group_times_kernel_plus_bias')
    if name=='linear':
        w=a.get('weight')
        if not (tensor(x) and tensor(w) and len(w['shape'])==2 and x['shape']
                and x['shape'][-1]==w['shape'][1] and shape==x['shape'][:-1]+[w['shape'][0]]
                and 'bias' in a and bias(a['bias'],w['shape'][0])):return row
        macs=n*w['shape'][1]
        return known(macs,2*macs+(n if tensor(a['bias']) else 0),0,'linear_dense_plus_bias')
    if name in ('mm','matmul','bmm','addmm','baddbmm','dot'):
        left=a.get('mat1',a.get('batch1',a.get('self',a.get('input'))))
        right=a.get('mat2',a.get('batch2',a.get('other',a.get('tensor'))))
        if not tensor(left) or not tensor(right):return row
        ls=left['shape'];rs=right['shape']
        if not ls or not rs:return row
        if name in ('mm','addmm') and (len(ls)!=2 or len(rs)!=2):return row
        if name in ('bmm','baddbmm') and (len(ls)!=3 or len(rs)!=3 or ls[0]!=rs[0]):return row
        if name=='dot' and (len(ls)!=1 or len(rs)!=1):return row
        k=ls[-1]
        if k!=(rs[-2] if len(rs)>1 else rs[0]):return row
        batch=_broadcast(ls[:-2] if len(ls)>1 else [],rs[:-2] if len(rs)>1 else [])
        if batch is None:return row
        expected=batch+(ls[-2:-1] if len(ls)>1 else [])+(rs[-1:] if len(rs)>1 else [])
        if shape!=expected:return row
        macs=n*k;flops=2*macs
        if name in ('addmm','baddbmm'):
            alpha=a.get('alpha');beta=a.get('beta');addend=a.get('self',a.get('input'))
            if not (_scalar(alpha) and _scalar(beta) and tensor(addend)
                    and _broadcast(addend['shape'],shape)==shape):return row
            flops+=n*(int(alpha!=1)+int(beta!=0)+int(beta not in (0,1)))
            if alpha!=1 or beta not in (0,1):return known(macs,flops,0,'dense_matmul_plus_alpha_beta_scaling_and_add')
        return known(macs,flops,0,'dense_matmul_plus_optional_beta1_add')
    if name.rstrip('_') in ('add','sub','mul','div'):
        other=a.get('other');base=name.rstrip('_')
        if not tensor(x) or not (tensor(other) or _scalar(other)):return row
        if _broadcast(x['shape'],other['shape'] if tensor(other) else [])!=shape:return row
        if base in ('add','sub') and not _scalar(a.get('alpha')):return row
        if base=='div' and a.get('rounding_mode') is not None:return row
        return known(0,n*(2 if base in ('add','sub') and a['alpha']!=1 else 1),0,
                     'broadcast_output_scalar_arithmetic')
    if name in ('relu','relu_','clamp_min'):
        if not tensor(x) or x['shape']!=shape:return row
        if name=='clamp_min' and not _scalar(a.get('min')):return row
        return known(0,0,n,'comparison_only_activation')
    if name in ('adaptive_avg_pool1d','adaptive_avg_pool2d','_adaptive_avg_pool2d'):
        d=1 if name.endswith('1d') else 2
        if not tensor(x) or len(shape) not in (d+1,d+2) or len(x['shape'])!=len(shape):return row
        if shape[:-d]!=x['shape'][:-d] or any(v<=0 for v in x['shape'][-d:]+shape[-d:]):return row
        sums=[sum(((i+1)*src+dst-1)//dst-(i*src)//dst for i in range(dst))
              for src,dst in zip(x['shape'][-d:],shape[-d:])]
        return known(0,math.prod(shape[:-d])*math.prod(sums),0,
                     'adaptive_bin_sum_k_minus_one_additions_and_one_division')
    if name in ('avg_pool1d','avg_pool2d','max_pool1d','max_pool2d','max_pool2d_with_indices'):
        d=1 if name.endswith('1d') else 2
        if not tensor(x) or len(shape) not in (d+1,d+2) or len(x['shape'])!=len(shape):return row
        kernel=a.get('kernel_size');stride=a.get('stride');padding=a.get('padding')
        dilation=a.get('dilation') if name.startswith('max') else [1]*d
        if not all(type(v) is list and len(v)==d and all(type(t) is int and t>=lo for t in v)
                   for v,lo in ((kernel,1),(padding,0),(dilation,1))):return row
        if stride==[]:stride=kernel
        if type(stride) is not list or len(stride)!=d or any(type(t) is not int or t<=0 for t in stride):return row
        if any(padding) or a.get('ceil_mode') is not False:return row
        expected=[(src-dil*(k-1)-1)//s+1 for src,k,s,dil in zip(x['shape'][-d:],kernel,stride,dilation)]
        if shape!=x['shape'][:-d]+expected:return row
        if name.startswith('avg') and a.get('divisor_override') is not None:return row
        k=math.prod(kernel)
        return known(0,n*k if name.startswith('avg') else 0,n*(k-1) if name.startswith('max') else 0,
                     'full_nonpadded_fixed_pool_windows')
    if name in ('native_batch_norm','_native_batch_norm_legit','_native_batch_norm_legit_no_training','batch_norm'):
        training=False if name=='_native_batch_norm_legit_no_training' else a.get('training')
        if training is not False:return dict(row,reason='batch_norm_training_statistics_not_modeled')
        if not tensor(x) or len(shape)<2 or shape!=x['shape'] or n==0:return row
        c=shape[1]
        if not all(tensor(a.get(key)) and a[key]['shape']==[c] for key in ('running_mean','running_var')):return row
        if not all(key in a and bias(a[key],c) for key in ('weight','bias')) or not _scalar(a.get('eps')) or a['eps']<0:return row
        # Unfused reference: sqrt(var+eps) per channel; subtract/divide per
        # element, optional multiply/add. A sqrt is ONE logical scalar op,
        # explicitly not an instruction count or hardware-equivalent latency.
        return known(0,2*c+n*(2+int(tensor(a['weight']))+int(tensor(a['bias']))),0,
                     'batch_norm_inference_unfused_2C_plus_N_times_2_plus_affine; sqrt=1_logical_op_not_hardware')
    if name in ('sum','mean'):
        if not tensor(x):return row
        if len(parts)>2 and parts[2].startswith('dim') and not all(key in a for key in ('dim','keepdim')):return row
        dims=a.get('dim',list(range(len(x['shape']))));rank=len(x['shape'])
        if dims==[]:dims=list(range(rank))
        if type(dims) is not list or any(type(d) is not int or not -rank<=d<rank for d in dims):return row
        dims=[d%rank for d in dims]
        if len(set(dims))!=len(dims) or type(a.get('keepdim',False)) is not bool:return row
        expected=[1 if i in dims else s for i,s in enumerate(x['shape'])] if a.get('keepdim',False) else [s for i,s in enumerate(x['shape']) if i not in dims]
        if shape!=expected:return row
        k=math.prod(x['shape'][d] for d in dims)
        if k==0:return dict(row,reason='empty_reduction_not_modeled')
        return known(0,n*(k-1+int(name=='mean')),0,'reduction_k_minus_one_additions_plus_mean_division')
    if name in ('silu','silu_'):
        return dict(row,reason='silu_exp_cost_has_no_arithmetic_flop_convention')
    if name=='dropout' and a.get('train') is False and tensor(x) and x['shape']==shape:
        return known(0,0,0,'inference_dropout_identity')
    return row


def summarize(calls,forwards,batch,*,model_version=2):
    if type(model_version) is not int or model_version not in (1,2,3):raise ValueError('unsupported operation model version')
    if type(forwards) is not int or forwards<=0 or type(batch) is not int or batch<=0:
        raise ValueError('invalid operation normalization')
    if model_version in (2,3):
        for i,c in enumerate(calls):
            if (type(c.get('call_index')) is not int or c['call_index']!=i
                    or type(c.get('measurement_index')) is not int or not 0<=c['measurement_index']<forwards
                    or type(c.get('operator')) is not str):raise ValueError('invalid operation association')
    rows=[cost(c,model_version=model_version) for c in calls];known=[r for r in rows if r['flops'] is not None]
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
    if model_version==3:
        for field in ('transcendental_ops','root_ops'):
            subtotal={}
            for row in known:
                for name,count in row[field].items():subtotal[name]=subtotal.get(name,0)+count
            m['supported_subtotal_'+field]=subtotal
            m[field+'_per_batch']=None if unknown else {k:v/forwards for k,v in subtotal.items()}
        m['flop_scope']='ordinary_arithmetic_only; special operation counts are not FLOP-equivalent'
        m['total_comparisons_per_batch']=None if unknown else comparisons/forwards
        m['comparisons_scope']='comparisons_per_batch is supported subtotal; total requires complete coverage'
    return rows,m


def _expanded_cost(call):
    """v3 operation vectors. v1/v2 derivations remain byte-compatible readers."""
    base=cost(call,model_version=2)
    row=dict(base,transcendental_ops=None,root_ops=None)
    a=call['arguments'];out=call['output']
    if type(a) is not dict or not _metadata(a) or not _metadata(out):return row
    if type(out) is list and out:out=out[0]
    if not _tensor(out) or out['dtype'] not in _FLOATS:
        return dict(row,transcendental_ops={},root_ops={}) if base['flops'] is not None else row
    parts=call['operator'].split('.');name=parts[1] if len(parts)>1 and parts[0]=='aten' else '<custom>'
    shape=out['shape'];n=_size(out);x=a.get('input',a.get('self'))
    def tensor(v):return _tensor(v) and (v['dtype'],v['device'])==(out['dtype'],out['device'])
    def known(flops,policy,*,macs=0,comparisons=0,trans=None,roots=None):
        return dict(row,flops=flops,macs=macs,comparisons=comparisons,status='modeled',reason=None,
                    policy=policy,transcendental_ops=trans or {},root_ops=roots or {})
    def unknown(reason):
        return dict(row,flops=None,macs=None,comparisons=None,status='unknown',policy=None,reason=reason)
    # A broadcast that expands self is invalid for an in-place operation.
    if name in ('add_','sub_','mul_','div_') and (not tensor(x) or x['shape']!=shape):
        return unknown('inplace_output_shape_mismatch')
    if name in ('conv_transpose1d','conv_transpose2d','conv_transpose3d'):
        a=dict(a,transposed=True)
    if name in ('convolution','_convolution','conv_transpose1d','conv_transpose2d','conv_transpose3d') and a.get('transposed') is True:
        w=a.get('weight');g=a.get('groups');d=len(shape)-2
        if not (d in (1,2,3) and tensor(x) and tensor(w) and len(x['shape'])==len(w['shape'])==len(shape)
                and type(g) is int and g>0 and all(v>0 for v in x['shape'][1:]+w['shape'])
                and x['shape'][1]==w['shape'][0] and x['shape'][1]%g==0
                and shape[:2]==[x['shape'][0],w['shape'][1]*g] and 'bias' in a):return row
        bias=a['bias'];stride=a.get('stride');pad=a.get('padding');dil=a.get('dilation');op=a.get('output_padding')
        if bias is not None and not (tensor(bias) and bias['shape']==[shape[1]]):return row
        if not all(type(v) is list and len(v)==d and all(type(t) is int and t>=lo for t in v)
                   for v,lo in ((stride,1),(pad,0),(dil,1),(op,0))):return row
        if any(o>=max(s,z) for o,s,z in zip(op,stride,dil)):return row
        expected=[(src-1)*s-2*p+z*(k-1)+o+1 for src,s,p,z,k,o in
                  zip(x['shape'][2:],stride,pad,dil,w['shape'][2:],op)]
        if shape[2:]!=expected or any(v<=0 for v in expected):return row
        # Count clipped contributions in O(sum(kernel sizes)), not O(input*kernel).
        pairs=[]
        for src,k,s,p,z,dst in zip(x['shape'][2:],w['shape'][2:],stride,pad,dil,shape[2:]):
            pairs.append(sum(max(0,min(src-1,(dst-1+p-j*z)//s)-max(0,-((j*z-p)//s))+1) for j in range(k)))
        macs=x['shape'][0]*x['shape'][1]*w['shape'][1]*math.prod(pairs)
        return known(2*macs+(n if bias is not None else 0),'transposed_clipped_scatter_reference_plus_bias',macs=macs)
    if 'batch_norm' in name and base['flops'] is not None:
        # v2 counted each sqrt as one FLOP. v3 reports it in a separate domain.
        return known(base['flops']-shape[1],'batch_norm_inference_unfused_arithmetic_plus_separate_sqrt',roots={'sqrt':shape[1]})
    if name=='adaptive_max_pool2d':
        if not tensor(x) or len(shape) not in (3,4) or len(x['shape'])!=len(shape):return row
        if (shape[:-2]!=x['shape'][:-2] or a.get('output_size')!=shape[-2:]
                or any(v<=0 for v in x['shape'][-2:]+shape[-2:])):return row
        sums=[sum(((i+1)*src+dst-1)//dst-(i*src)//dst for i in range(dst))
              for src,dst in zip(x['shape'][-2:],shape[-2:])]
        return known(0,'adaptive_max_exact_bins_K_minus_one_comparisons',
                     comparisons=math.prod(shape[:-2])*math.prod(sums)-n)
    if name=='max' and len(parts)>2 and parts[2]=='dim':
        dim=a.get('dim');keep=a.get('keepdim')
        if not (tensor(x) and x['shape'] and type(dim) is int and -len(x['shape'])<=dim<len(x['shape'])
                and type(keep) is bool):return row
        dim%=len(x['shape']);expected=list(x['shape'])
        if keep:expected[dim]=1
        else:expected.pop(dim)
        if shape!=expected or x['shape'][dim]<=0:return row
        return known(0,'max_reduction_K_minus_one_comparisons',comparisons=n*(x['shape'][dim]-1))
    if name=='hardswish':
        if not tensor(x) or x['shape']!=shape:return row
        return known(3*n,'hardswish_x_times_clamp_x_plus_3_divide_6',comparisons=2*n)
    if name in ('native_layer_norm','native_group_norm','layer_norm','group_norm'):
        if not tensor(x) or x['shape']!=shape or not n or not _scalar(a.get('eps')) or a['eps']<0:return row
        if name in ('native_layer_norm','layer_norm'):
            dims=a.get('normalized_shape')
            if not (type(dims) is list and dims and all(type(s) is int and s>0 for s in dims)
                    and len(dims)<=len(shape) and shape[-len(dims):]==dims):return row
            groups=n//math.prod(dims);affine_shape=dims
        else:
            g=a.get('group',a.get('num_groups'))
            if not (len(shape)>=2 and type(g) is int and g>0 and shape[1]%g==0):return row
            if name=='native_group_norm' and not (all(type(a.get(key)) is int for key in ('N','C','HxW'))
                    and [a['N'],a['C'],a['HxW']]==[shape[0],shape[1],math.prod(shape[2:])]):return row
            groups=shape[0]*g;affine_shape=[shape[1]]
        if not all(key in a and (a[key] is None or tensor(a[key]) and a[key]['shape']==affine_shape) for key in ('weight','bias')):return row
        # Mean K; population variance 3K; epsilon 1; centering+scaling 2K.
        affine=int(a['weight'] is not None)+int(a['bias'] is not None)
        return known(n*(6+affine)+groups,'population_mean_variance_unfused_reference; rsqrt_separate',roots={'rsqrt':groups})
    if name in ('_softmax','_log_softmax','softmax','log_softmax'):
        dim=a.get('dim')
        if not (tensor(x) and x['shape']==shape and shape and type(dim) is int
                and -len(shape)<=dim<len(shape)):return row
        if name.startswith('_'):
            if a.get('half_to_float') is not False:return row
        elif 'dtype' not in a or a['dtype'] not in (None,out['dtype']):return row
        k=shape[dim]
        if k<=0:return row
        groups=n//k;reductions=groups*(k-1)
        trans={'exp':n}
        if name in ('_log_softmax','log_softmax'):trans['log']=groups
        return known(2*n+reductions,'stable_max_shift_exp_sum_'+('log_subtract' if name in ('_log_softmax','log_softmax') else 'divide'),
                     comparisons=reductions,trans=trans)
    if name in ('sigmoid','sigmoid_','silu','silu_','exp','exp_','log','log_','tanh','tanh_','gelu'):
        if not tensor(x) or x['shape']!=shape:return row
        if name.rstrip('_') in ('exp','log','tanh'):
            return known(0,'named_transcendental_not_weighted_as_FLOPs',trans={name.rstrip('_'):n})
        if name.startswith(('sigmoid','silu')):
            return known(n*(4 if name.startswith('silu') else 3),'negate_exp_add_divide'+('_multiply' if name.startswith('silu') else ''),trans={'exp':n})
        if a.get('approximate')=='none':return known(4*n,'gelu_erf_reference_precomputed_constants',trans={'erf':n})
        if a.get('approximate')=='tanh':return known(8*n,'gelu_tanh_reference_cubic_as_two_multiplies',trans={'tanh':n})
        return row
    if base['flops'] is not None:
        return dict(row,transcendental_ops={},root_ops={})
    return row
