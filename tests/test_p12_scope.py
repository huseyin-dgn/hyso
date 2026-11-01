"""Independent arithmetic, native CPU dispatch and legacy P12 compatibility."""
import collections
import copy
import itertools
import json
import math
import os
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from hyso.performance._operation_models import cost, summarize, CONVENTION
from hyso.performance._profile_worker import _meta


def tensor(shape):return dict(kind='tensor',shape=list(shape),dtype='torch.float32',device='cpu')


def call(name, arguments, shape):
    return dict(operator='aten.'+name, arguments=arguments, output=tensor(shape),call_index=0,measurement_index=0)


def current(c):return cost(c,model_version=3)


def observe(model,x):
    calls=[]
    class Observer(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            kwargs=kwargs or {};out=func(*args,**kwargs);bound={}
            for i,arg in enumerate(func._schema.arguments):
                if i<len(args):bound[arg.name]=_meta(args[i])
                elif arg.name in kwargs:bound[arg.name]=_meta(kwargs[arg.name])
                elif arg.has_default_value():bound[arg.name]=_meta(arg.default_value)
            calls.append(dict(call_index=len(calls),measurement_index=0,operator=str(func),arguments=bound,output=_meta(out)))
            return out
    with torch.inference_mode(), Observer():model(x)
    return calls


@pytest.mark.parametrize('d,groups,padding,stride,dilation,op',[
    (1,1,0,1,1,0),(1,2,1,2,1,1),(2,2,1,2,1,1),
    (2,4,2,2,2,1),(3,2,1,1,1,0)])
@pytest.mark.parametrize('bias',[False,True])
def test_transposed_clipped_contributions_bruteforce(d,groups,padding,stride,dilation,op,bias):
    src=[3]*d;kernel=[3]*d;dst=[(v-1)*stride-2*padding+dilation*2+op+1 for v in src]
    shape=[2,8]+dst
    a=dict(input=tensor([2,4]+src),weight=tensor([4,8//groups]+kernel),bias=tensor([8]) if bias else None,
           groups=groups,stride=[stride]*d,padding=[padding]*d,dilation=[dilation]*d,output_padding=[op]*d,transposed=True)
    c=call('convolution.default',a,shape)
    # Independent enumeration of all spatial input/kernel pairs and clipping.
    pairs=sum(all(0<=i*stride-padding+k*dilation<out for i,k,out in zip(xs,ks,dst))
              for xs in itertools.product(*(range(s) for s in src))
              for ks in itertools.product(*(range(k) for k in kernel)))
    expected=2*4*(8//groups)*pairs
    r=current(c)
    assert r['macs']==expected and r['flops']==2*expected+math.prod(shape)*bias
    assert cost(c,model_version=2)['flops'] is None
    for key,value in [('groups',3),('stride',[0]*d),('output_padding',[stride+dilation]*d)]:
        broken=copy.deepcopy(c);broken['arguments'][key]=value
        assert current(broken)['flops'] is None


@pytest.mark.parametrize('weight,bias',[(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize('name',['native_layer_norm','native_group_norm'])
def test_normalization_reference(name,weight,bias):
    shape=[2,4,3];n=24
    if name=='native_layer_norm':
        a=dict(input=tensor(shape),normalized_shape=[4,3],eps=1e-5);affine=[4,3];groups=2
    else:
        a=dict(input=tensor(shape),N=2,C=4,HxW=3,group=2,eps=1e-5);affine=[4];groups=4
    a.update(weight=tensor(affine) if weight else None,bias=tensor(affine) if bias else None)
    r=current(call(name+'.default',a,shape))
    # Per group: mean K; var K sub + K square + (K-1) add + 1 div;
    # epsilon 1; output K sub + K mul; optional affine K mul/add.
    k=n//groups
    expected=groups*((k-1)+1+k+k+(k-1)+1+1+k+k+k*weight+k*bias)
    assert r['flops']==expected and r['root_ops']=={'rsqrt':groups}
    assert r['transcendental_ops']=={}
    a['eps']=-1
    assert current(call(name+'.default',a,shape))['flops'] is None


@pytest.mark.parametrize('name',['_softmax','_log_softmax'])
@pytest.mark.parametrize('dim',[0,1,-1])
def test_softmax_reference(name,dim):
    shape=[2,3,4];n=24;k=shape[dim];groups=n//k
    r=current(call(name+'.default',dict(self=tensor(shape),dim=dim,half_to_float=False),shape))
    assert r['flops']==2*n+groups*(k-1)
    assert r['comparisons']==groups*(k-1)
    assert r['transcendental_ops']==dict(exp=n,**({'log':groups} if name=='_log_softmax' else {}))
    assert current(call(name+'.default',dict(self=tensor(shape),dim=dim,half_to_float=True),shape))['flops'] is None


@pytest.mark.parametrize('name,factor,special',[
    ('sigmoid',3,'exp'),('silu',4,'exp'),('exp',0,'exp'),('log',0,'log'),('tanh',0,'tanh')])
def test_activation_vectors(name,factor,special):
    r=current(call(name+'.default',dict(self=tensor([2,3])),[2,3]))
    assert r['flops']==6*factor and r['transcendental_ops']=={special:6}


def test_comparison_only_reductions_and_hardswish():
    a=dict(self=tensor([2,3,5,7]),output_size=[2,3])
    r=current(call('adaptive_max_pool2d.default',a,[2,3,2,3]))
    # Independent explicit bins, including overlapping boundaries.
    expected=sum((math.ceil((i+1)*5/2)-math.floor(i*5/2))*(math.ceil((j+1)*7/3)-math.floor(j*7/3))-1
                 for i in range(2) for j in range(3))*6
    assert r['flops']==0 and r['comparisons']==expected
    r=current(call('max.dim',dict(self=tensor([2,3,4]),dim=1,keepdim=False),[2,4]))
    assert r['flops']==0 and r['comparisons']==2*4*(3-1)
    r=current(call('hardswish.default',dict(self=tensor([2,3,4])),[2,3,4]))
    assert r['flops']==3*24 and r['comparisons']==2*24


def test_v3_matrix_scalars_broadcast_and_depthwise():
    a=dict(self=tensor([5]),mat1=tensor([2,3]),mat2=tensor([3,5]),alpha=-2,beta=3)
    r=current(call('addmm.default',a,[2,5]))
    assert r['macs']==30 and r['flops']==60+10*3
    r=current(call('matmul.default',dict(self=tensor([2,1,3,4]),other=tensor([5,4,6])),[2,5,3,6]))
    assert r['macs']==2*5*3*6*4
    a=dict(input=tensor([2,4,5,7]),weight=tensor([8,1,3,3]),groups=4,bias=tensor([8]),
           stride=[1,1],padding=[1,1],dilation=[1,1],transposed=False)
    r=current(call('convolution.default',a,[2,8,5,7]))
    assert r['macs']==2*8*5*7*9 and r['flops']==2*r['macs']+2*8*5*7


@pytest.mark.parametrize('approximate,factor,special',[('none',4,'erf'),('tanh',8,'tanh')])
def test_gelu_reference(approximate,factor,special):
    r=current(call('gelu.default',dict(self=tensor([2,3]),approximate=approximate),[2,3]))
    assert r['flops']==6*factor and r['transcendental_ops']=={special:6}


@pytest.mark.parametrize('forwards,batch',[(0,2),(-1,2),(True,2),(1,0),(1,-1),(1,True),(1,1.5)])
@pytest.mark.parametrize('version',[1,2,3])
def test_invalid_normalization_rejected(forwards,batch,version):
    with pytest.raises(ValueError):summarize([],forwards,batch,model_version=version)


def test_unknowns_poison_totals_and_keep_symbolic_subtotals():
    a=call('silu.default',dict(self=tensor([2,3])),[2,3])
    b=call('sin.default',dict(self=tensor([2,3])),[2,3]);b['call_index']=1
    rows,m=summarize([a,b],1,2,model_version=3)
    assert rows[1]['flops'] is None and m['total_flops_per_batch'] is None
    assert m['transcendental_ops_per_batch'] is None and m['supported_subtotal_transcendental_ops']=={'exp':6}
    assert m['supported_subtotal_flops']==24 and m['status']=='partial'
    b['operator']='custom.silu.default'
    assert current(b)['flops'] is None
    bad=call('add_.Tensor',dict(self=tensor([1,3]),other=tensor([2,3]),alpha=1),[2,3])
    assert current(bad)['flops'] is None


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_native_cpu_fixture_inventory_and_independent_vectors(dtype):
    from profiling_factories import Norm,Shared,Matrix
    from hyso.models.modular import ModularCNN
    models={
        'Norm':(Norm(),torch.ones(2,2,5,7)),
        'Shared':(Shared(),torch.ones(2,2,5,7)),
        'Matrix':(Matrix(),torch.ones(2,1,5,7)),
        'LayerNorm':(nn.LayerNorm([4,3,5]),torch.ones(2,4,3,5)),
        'GroupNorm':(nn.GroupNorm(2,4),torch.ones(2,4,3,5)),
        'Softmax':(nn.Softmax(dim=1),torch.ones(2,4,3,5)),
        'LogSoftmax':(nn.LogSoftmax(dim=-1),torch.ones(2,4,3,5)),
        'GELU':(nn.GELU(),torch.ones(2,4,3,5)),
        'Grouped':(nn.Conv2d(4,8,3,padding=1,groups=2),torch.ones(2,4,3,5)),
        'Depthwise':(nn.Conv2d(4,4,3,padding=1,groups=4),torch.ones(2,4,3,5)),
        'Transpose':(nn.ConvTranspose2d(4,8,3,stride=2,padding=1,output_padding=1,groups=2),torch.ones(2,4,3,5))}
    for attention in ('none','se','cbam','eca','coordinate'):
        models['ModularCNN-'+attention]=(ModularCNN(input_channels=1,num_classes=3,widths=(4,),repeats=(1,),strides=(1,),attention=attention),torch.ones(2,1,5,7))
    before=collections.Counter();after=collections.Counter();records={}
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        for name,(model,x) in models.items():
            calls=observe(model.to(dtype=dtype).eval(),x.to(dtype));rows,m=summarize(calls,1,len(x),model_version=3)
            for c,row in zip(calls,rows):
                if cost(c,model_version=2)['flops'] is None:before[c['operator']]+=1
                if row['flops'] is None:after[c['operator']]+=1
            records[name]=dict(calls=calls,measurements=m)
    destination=os.environ.get('HYSO_P12_EVIDENCE')
    if destination:
        path=Path(destination);path.mkdir(parents=True,exist_ok=True)
        (path/('native-dispatch-inventory-'+str(dtype).split('.')[-1]+'.json')).write_text(json.dumps(dict(
            unknown_before=before.most_common(),unknown_after=after.most_common(),models=records),indent=2))
    assert any(e.key=='aten::native_layer_norm' for e in prof.key_averages())
    assert records['Norm']['measurements']['status']=='complete'
    assert records['Softmax']['measurements']['total_flops_per_batch']==2*120+30*3
    assert records['Softmax']['measurements']['transcendental_ops_per_batch']=={'exp':120}
    assert records['ModularCNN-eca']['measurements']['status']=='complete'
    assert records['ModularCNN-cbam']['measurements']['status']=='complete'
    assert records['ModularCNN-coordinate']['measurements']['status']=='complete'
    assert records['Depthwise']['measurements']['total_macs_per_batch']==2*4*3*5*9
    assert records['Grouped']['measurements']['total_macs_per_batch']==2*8*3*5*2*9
    assert not torch.cuda.is_initialized()


def test_public_api_v1_v2_v3_reopen_and_symbolic_tamper(tmp_path):
    import csv
    from hyso.performance import ColdStartRecipe,ProfilingProtocol,profile_operators,analyze_flops,OperationCountResult
    source=profile_operators(ColdStartRecipe('profiling_factories:Norm',factory_search_path=Path(__file__).parent,input_shape=(2,2,5,7)),
        protocol=ProfilingProtocol(warmup_iterations=0,measure_iterations=1,timeout_seconds=45))
    result=analyze_flops(source);data=result.to_dict()
    assert data['protocol']['operation_model_version']==3 and data['schema_version']==1
    result.to_json(tmp_path/'new.json');result.to_csv(tmp_path/'new.csv')
    assert OperationCountResult(json.loads((tmp_path/'new.json').read_text())).to_dict()==data
    payloads=[json.loads(row['payload']) for row in csv.DictReader((tmp_path/'new.csv').open())]
    assert any(row['transcendental_ops'] for row in payloads)
    for version in (1,2):
        old=copy.deepcopy(data);old['protocol']['operation_model_version']=version;old['protocol']['convention']=CONVENTION
        old['raw_evidence']['operations'],old['measurements']=summarize(old['raw_evidence']['aten_calls'],1,2,model_version=version)
        assert OperationCountResult(old).to_dict()==old
    bad=copy.deepcopy(data)
    next(row for row in bad['raw_evidence']['operations'] if row['transcendental_ops'])['transcendental_ops']['exp']=0
    with pytest.raises(ValueError):OperationCountResult(bad)
