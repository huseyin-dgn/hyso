"""Independent small-shape logical arithmetic references; no tensor allocation."""
import copy
import math

import pytest

from hyso.performance._operation_models import cost,summarize


def tensor(shape,dtype='torch.float32'):
    return dict(kind='tensor',shape=list(shape),dtype=dtype,device='cpu')


def call(name,arguments,shape,**kwargs):
    return dict(operator='aten.'+name,arguments=arguments,output=tensor(shape),
                call_index=0,measurement_index=0,**kwargs)


@pytest.mark.parametrize('dtype',['torch.float32','torch.float64'])
@pytest.mark.parametrize('weight,bias',[(False,False),(True,False),(False,True),(True,True)])
def test_bn_unfused_reference(dtype,weight,bias):
    shape=(2,3,4);c=shape[1];n=math.prod(shape)
    a=dict(input=tensor(shape,dtype),weight=tensor((c,),dtype) if weight else None,
        bias=tensor((c,),dtype) if bias else None,running_mean=tensor((c,),dtype),
        running_var=tensor((c,),dtype),training=False,momentum=.1,eps=1e-5)
    observed=call('native_batch_norm.default',a,shape);observed['output']=tensor(shape,dtype)
    r=cost(observed)
    # C additions + C square roots; N subtracts/divides; optional affine.
    assert r['flops']==c+c+n+n+n*weight+n*bias
    assert r['macs']==0 and 'sqrt=1_logical_op_not_hardware' in r['policy']
    a['training']=True
    assert cost(observed)['flops'] is None
    for key in ('running_mean','running_var','weight','bias','training','eps'):
        broken=copy.deepcopy(observed);broken['arguments'].pop(key)
        assert cost(broken)['flops'] is None


@pytest.mark.parametrize('alpha,beta',[(1,1),(1,0),(2,1),(-2,3),(0,0),(1,-1)])
@pytest.mark.parametrize('batched',[False,True])
def test_matrix_alpha_beta_reference(alpha,beta,batched):
    batch=[2] if batched else [];m,k,n=3,4,5;shape=batch+[m,n]
    a=dict(self=tensor((n,)),alpha=alpha,beta=beta)
    a.update(dict(batch1=tensor(batch+[m,k]),batch2=tensor(batch+[k,n])) if batched else
             dict(mat1=tensor((m,k)),mat2=tensor((k,n))))
    observed=call('baddbmm.default' if batched else 'addmm.default',a,shape)
    r=cost(observed);outputs=math.prod(shape)
    assert r['macs']==outputs*k
    assert r['flops']==2*outputs*k+outputs*(int(alpha!=1)+int(beta!=0)+int(beta not in (0,1)))
    for key in ('batch2' if batched else 'mat2','self','alpha','beta'):
        broken=copy.deepcopy(observed);broken['arguments'].pop(key)
        assert cost(broken)['flops'] is None


@pytest.mark.parametrize('left,right,output,k',[
    ([3],[3],[],3),([2,3],[3],[2],3),([3],[2,3,4],[2,4],3),
    ([2,1,3,4],[5,4,6],[2,5,3,6],4),([0,3],[3,2],[0,2],3)])
def test_matmul_vector_and_broadcast(left,right,output,k):
    r=cost(call('matmul.default',dict(self=tensor(left),other=tensor(right)),output))
    assert r['macs']==math.prod(output)*k and r['flops']==2*math.prod(output)*k


@pytest.mark.parametrize('name,alpha,factor',[('add',1,1),('sub',2,2),('mul',1,1),('div',1,1)])
def test_elementwise_broadcast(name,alpha,factor):
    a=dict(self=tensor((2,1,3)),other=tensor((4,1)),alpha=alpha)
    observed=call(name+'.Tensor',a,(2,4,3))
    assert cost(observed)['flops']==2*4*3*factor
    a['other']=2
    observed['output']=tensor((2,1,3))
    assert cost(observed)['flops']==2*1*3*factor
    a.pop('other')
    assert cost(observed)['flops'] is None


@pytest.mark.parametrize('name',['sum','mean'])
@pytest.mark.parametrize('dims,keepdim,output',[
    ([-1],False,[2,3]),([1,2],True,[2,1,1]),([],False,[]),([0,2],False,[3])])
def test_reduction_formula(name,dims,keepdim,output):
    a=dict(self=tensor((2,3,4)),dim=dims,keepdim=keepdim)
    observed=call(name+'.dim_IntList',a,output);n=math.prod(output);k=24//n
    assert cost(observed)['flops']==n*(k-1+int(name=='mean'))
    for key in ('dim','keepdim','self'):
        broken=copy.deepcopy(observed);broken['arguments'].pop(key)
        assert cost(broken)['flops'] is None
    a['dim']=[1,1]
    assert cost(observed)['flops'] is None


@pytest.mark.parametrize('name',['sum','mean'])
def test_empty_reduction_unknown(name):
    assert cost(call(name+'.dim_IntList',dict(self=tensor((2,0)),dim=[1],keepdim=False),(2,)))['flops'] is None


@pytest.mark.parametrize('shape,output',[((2,3,5,7),(2,3,2,3)),((3,5,7),(3,2,3))])
def test_adaptive_pool_overlapping_bins_and_unbatched(shape,output):
    windows=[sum(math.ceil((i+1)*src/dst)-math.floor(i*src/dst) for i in range(dst))
             for src,dst in zip(shape[-2:],output[-2:])]
    r=cost(call('_adaptive_avg_pool2d.default',dict(self=tensor(shape),output_size=[2,3]),output))
    assert r['flops']==math.prod(shape[:-2])*math.prod(windows)


def test_fixed_pool_boundaries_and_dilation():
    a=dict(self=tensor((2,3,7,7)),kernel_size=[2,2],stride=[2,2],padding=[0,0],dilation=[2,2],ceil_mode=False)
    observed=call('max_pool2d_with_indices.default',a,(2,3,3,3))
    assert cost(observed)['comparisons']==2*3*3*3*(2*2-1)
    for key,value in [('ceil_mode',True),('padding',[1,1]),('stride',None),('kernel_size',[])]:
        broken=copy.deepcopy(observed);broken['arguments'][key]=value
        assert cost(broken)['flops'] is None
    observed['output']=tensor((2,3,4,4))
    assert cost(observed)['flops'] is None
    observed['output']=tensor((2,3,3,3));a.pop('dilation')
    assert cost(observed)['flops'] is None


@pytest.mark.parametrize('groups,bias',[(1,False),(2,True),(4,True)])
def test_grouped_convolution_formula_and_missing_bias(groups,bias):
    a=dict(input=tensor((2,4,5,7)),weight=tensor((8,4//groups,3,3)),
        bias=tensor((8,)) if bias else None,groups=groups,stride=[1,1],padding=[1,1],
        dilation=[1,1],transposed=False)
    observed=call('convolution.default',a,(2,8,5,7));outputs=2*8*5*7
    assert cost(observed)['flops']==2*outputs*(4//groups)*3*3+outputs*bias
    a.pop('bias')
    assert cost(observed)['flops'] is None


@pytest.mark.parametrize('bad',[{},None,{'kind':'tensor'},tensor((-1,3)),tensor((True,3)),
    tensor((2,3),'torch.int64'),dict(kind='unrecorded'),float('nan')])
def test_missing_or_invalid_operand_metadata_is_unknown(bad):
    observed=call('mm.default',dict(self=tensor((2,3)),mat2=bad),(2,4))
    # mm uses 'mat2', while its first argument is self.
    assert cost(observed)['flops'] is None


def test_dtype_shape_scalar_namespace_and_inplace_rejection():
    observed=call('add.Tensor',dict(self=tensor((2,3)),other=tensor((3,)),alpha=1),(2,3))
    for key,value in [('alpha',{}),('alpha',float('inf')),('other',tensor((4,))),('self',tensor((2,3),'torch.float64'))]:
        broken=copy.deepcopy(observed);broken['arguments'][key]=value
        assert cost(broken)['flops'] is None
    observed['operator']='custom.add.Tensor'
    assert cost(observed)['flops'] is None
    observed['operator']='aten.silu.default'
    assert cost(observed)['flops'] is None and 'exp' in cost(observed)['reason']


def test_per_forward_mean_unknown_subtotal_and_ledger():
    a=call('mul.Tensor',dict(self=tensor((3,2)),other=2),(3,2))
    b=call('mul.Tensor',dict(self=tensor((3,4)),other=2),(3,4));b.update(call_index=1,measurement_index=1)
    _,m=summarize([a,b],2,3)
    assert m['total_flops_per_batch']==(3*2+3*4)/2 and m['total_flops_per_sample']==3
    b['operator']='custom.mul.Tensor'
    rows,m=summarize([a,b],2,3)
    assert rows[1]['flops'] is None and m['total_flops_per_batch'] is None
    assert m['supported_subtotal_flops']==6 and m['supported_flops_per_batch']==3
    b['measurement_index']=2
    with pytest.raises(ValueError):summarize([a,b],2,3)
    with pytest.raises(ValueError):summarize([a],0,3)


def test_legacy_model_reader_preserves_unknowns():
    observed=call('mean.dim',dict(self=tensor((2,3)),dim=[1],keepdim=False),(2,))
    assert cost(observed,model_version=1)['flops'] is None
    assert cost(observed)['flops']==2*3


@pytest.mark.parametrize('dtype_name',['float32','float64'])
def test_real_cpu_dispatch_training_transcendentals_and_scalar_models(dtype_name):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from hyso.performance._profile_worker import _meta
    calls=[]
    class Observer(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            kwargs=kwargs or {};output=func(*args,**kwargs);bound={}
            for i,arg in enumerate(func._schema.arguments):
                if i<len(args):bound[arg.name]=_meta(args[i])
                elif arg.name in kwargs:bound[arg.name]=_meta(kwargs[arg.name])
                elif arg.has_default_value():bound[arg.name]=_meta(arg.default_value)
            calls.append(dict(call_index=len(calls),measurement_index=0,operator=str(func),
                              arguments=bound,output=_meta(output)))
            return output
    dtype=getattr(torch,dtype_name)
    left=torch.ones(2,3,dtype=dtype);right=torch.ones(3,4,dtype=dtype);addend=torch.ones(4,dtype=dtype)
    x=torch.ones(2,3,4,dtype=dtype);bn=torch.nn.BatchNorm1d(3).to(dtype=dtype).train()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with Observer():
            torch.addmm(addend,left,right,alpha=2,beta=-3)
            torch.add(x,torch.ones(1,3,1,dtype=dtype),alpha=2)
            torch.nn.functional.silu(x)
            bn(x)
    rows=[cost(c) for c in calls]
    matrix=next(r for r in rows if r['operator'].startswith('aten.addmm.'))
    assert matrix['macs']==2*4*3 and matrix['flops']==2*(2*4*3)+3*(2*4)
    broadcast=next(r for r in rows if r['operator'].startswith('aten.add.Tensor'))
    assert broadcast['flops']==2*2*3*4
    nonlinear=next(r for r in rows if r['operator'].startswith('aten.silu.'))
    assert nonlinear['flops'] is None and 'exp' in nonlinear['reason']
    training=next(r for r in rows if 'batch_norm' in r['operator'])
    assert training['flops'] is None and training['reason']=='batch_norm_training_statistics_not_modeled'
    assert any(e.key=='aten::addmm' for e in prof.key_averages())
