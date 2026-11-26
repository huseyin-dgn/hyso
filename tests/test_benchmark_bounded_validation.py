"""Bounded CPU regressions: no GPU allocation, OOM workload or acceptance launch."""
import copy
from pathlib import Path
import pytest
import torch
from torch import nn
from hyso.benchmark import runner as r


@pytest.mark.parametrize('host',[False,True])
@pytest.mark.parametrize('position',[0,6,7,13,14,22])
@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_at_chunk_boundaries_and_tail(monkeypatch,host,position,bad):
    monkeypatch.setattr(r,'_FINITE_CHUNK_ELEMENTS',7)
    tensor=torch.ones(23);tensor[position]=bad
    assert r._all_finite(tensor,host=host) is False


@pytest.mark.parametrize('kind',['transpose','slice','expand','channels_last','offset'])
def test_strided_validation_and_host_copy_use_bounded_views(monkeypatch,kind):
    monkeypatch.setattr(r,'_FINITE_CHUNK_ELEMENTS',7)
    if kind=='transpose':value=torch.arange(60,dtype=torch.float32).reshape(6,10).t()
    elif kind=='slice':value=torch.arange(140,dtype=torch.float32).reshape(10,14)[::2,1::2]
    elif kind=='expand':value=torch.arange(5,dtype=torch.float32).reshape(1,5).expand(17,5)
    elif kind=='channels_last':value=torch.arange(120,dtype=torch.float32).reshape(2,3,4,5).to(memory_format=torch.channels_last)
    else:value=torch.arange(121,dtype=torch.float32)[1::2].reshape(6,10)
    signature=r._signature(value);expected=value.clone();sizes=[];copies=[]
    original=torch.isfinite;copy_=torch.Tensor.copy_
    def finite(chunk):
        sizes.append(chunk.numel());assert chunk.numel()<=7
        assert chunk.untyped_storage()._cdata==value.untyped_storage()._cdata
        return original(chunk)
    def transfer(dest,source,*a,**kw):
        copies.append(source.numel());assert source.numel()<=7 and dest.numel()<=7
        return copy_(dest,source,*a,**kw)
    def forbidden(*a,**kw):raise AssertionError('whole-tensor contiguous/reshape conversion')
    with monkeypatch.context() as m:
        m.setattr(torch,'isfinite',finite);m.setattr(torch.Tensor,'copy_',transfer)
        m.setattr(torch.Tensor,'contiguous',forbidden);m.setattr(torch.Tensor,'reshape',forbidden)
        assert r._all_finite(value)
        saved=r._host_copy(value)
    assert sum(sizes)==value.numel() and sum(copies)==value.numel()
    assert torch.equal(saved,expected) and r._signature(value)==signature
    value[tuple(n-1 for n in value.shape)]=float('nan')
    assert not r._all_finite(value)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32,torch.float64,torch.int64,torch.bool])
def test_scalar_empty_and_dtype_equivalence(dtype):
    for shape in ((),(0,),(3,5)):
        value=torch.ones(shape,dtype=dtype)
        assert r._all_finite(value)==torch.isfinite(value).all().item()
        saved=r._host_copy(value)
        assert saved.dtype==value.dtype and saved.shape==value.shape and torch.equal(saved,value)


def test_scratch_bound_and_complete_large_tensor_scan(monkeypatch):
    value=torch.ones(r._FINITE_CHUNK_ELEMENTS*2+1,dtype=torch.float64)
    calls=[];original=torch.isfinite
    def finite(chunk):calls.append(chunk.numel());return original(chunk)
    monkeypatch.setattr(torch,'isfinite',finite)
    assert r._all_finite(value) and sum(calls)==value.numel() and max(calls)<=r._FINITE_CHUNK_ELEMENTS
    assert r._FINITE_VALIDATION_SCRATCH_BYTES>=r._FINITE_CHUNK_ELEMENTS*(3*8+4)+2*1024**2
    assert r._FINITE_VALIDATION_SCRATCH_BYTES<=8*1024**2


@pytest.mark.parametrize('kind',['contiguous','transpose','slice','expand'])
def test_canonical_descriptor_and_bit_checks_preserve_byte_order(monkeypatch,kind):
    import hashlib
    monkeypatch.setattr(r,'_FINITE_CHUNK_ELEMENTS',7)
    value=torch.arange(60,dtype=torch.float32).reshape(6,10)
    if kind=='transpose':value=value.t()
    elif kind=='slice':value=value[:,::2]
    elif kind=='expand':value=value[:1].expand(9,10)
    model=nn.Module();model.register_buffer('values',value)
    modules,entries=r._profile(model,torch.float32)
    old=hashlib.sha256();old.update(str(('buffer','values',tuple(value.shape),str(value.dtype))).encode())
    old.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    assert r._descriptor(modules,entries)[1]==old.hexdigest()
    saved=r._host_copy(value)
    assert r._same_bits(value,saved)
    saved[tuple(n-1 for n in saved.shape)]=-1
    assert not r._same_bits(value,saved)


def test_output_validation_is_complete_and_keeps_contract(monkeypatch):
    monkeypatch.setattr(r,'_FINITE_CHUNK_ELEMENTS',7)
    value=torch.ones(11,3).t();expected=r._output(value,3)
    assert expected==dict(shape=[3,11],dtype='torch.float32',device='cpu')
    value[-1,-1]=float('inf')
    with pytest.raises(ValueError,match='finite'):r._output(value,3)
    with pytest.raises(ValueError):r._output(torch.ones(2,11),3)
    with pytest.raises(ValueError):r._output(torch.ones(3,11,dtype=torch.float64),3,expected)


def test_parameter_validation_keeps_dtype_device_and_input_contract(monkeypatch):
    monkeypatch.setattr(r,'_FINITE_CHUNK_ELEMENTS',7)
    parameter=nn.Parameter(torch.ones(4,11).t())
    r._tensor(parameter,torch.float32,parameter=True)
    with torch.no_grad():parameter[-1,-1]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):r._tensor(parameter,torch.float32,parameter=True)
    with pytest.raises(ValueError,match='dtype'):r._tensor(torch.ones(3,dtype=torch.float64),torch.float32)
    with pytest.raises(ValueError,match='device'):r._tensor(torch.ones(3),torch.float32,device=torch.device('meta'))
    with pytest.raises(ValueError,match='contiguous'):r._tensor(torch.ones(3,5).t(),torch.float32,input_tensor=True)


def test_snapshot_retains_signed_zero_nan_gradient_bits_and_alias_metadata(monkeypatch):
    monkeypatch.setattr(r,'_FINITE_CHUNK_ELEMENTS',7)
    model=nn.Linear(11,4);model.register_buffer('strided',torch.arange(44,dtype=torch.float32).reshape(4,11).t())
    model.train();model.weight.grad=torch.zeros_like(model.weight);model.weight.grad[0,0]=-0.;model.weight.grad[-1,-1]=float('nan')
    modules,entries=r._profile(model,torch.float32);x=torch.ones(2,11)
    snapshot=r._Snapshot(modules,entries,x)
    gradient=model.weight.grad;model.eval();snapshot.check(full=True)
    before=r._descriptor(modules,entries)
    snapshot.restore_modes()
    assert model.training and model.weight.grad is gradient
    assert before==r._descriptor(modules,entries) and torch.signbit(model.weight.grad[0,0]) and torch.isnan(model.weight.grad[-1,-1])


def test_snapshot_does_not_hide_resource_failure(monkeypatch):
    snapshot=r._Snapshot([],[],torch.ones(1));failure=torch.OutOfMemoryError('offline injected exception; no allocation')
    def fail(**kw):raise failure
    monkeypatch.setattr(snapshot,'_check',fail)
    with pytest.raises(torch.OutOfMemoryError) as error:snapshot.check(full=True)
    assert error.value is failure


def test_gpu_recovery_order_requires_verified_cleanup_and_fresh_admission():
    from tools.user_acceptance import runner as u
    p={'source_impact_review':{'kind':'BOUNDED_BENCHMARK_FINITE_VALIDATION','verified_oom_cleanup':True,'fresh_worker_admission_required':True}}
    cp={'safety_hold':{'features':['U02']}}
    order=u.gpu_group_order(p,cp)
    assert order.index('03-sweeps')<order.index('02-inference') and order.index('05-training')<order.index('02-inference')
    assert set(order)==set(u.GROUPS)
    bad=copy.deepcopy(p);bad['source_impact_review']['verified_oom_cleanup']=False
    assert u.gpu_group_order(bad,cp)==list(u.GROUPS)
    bad=copy.deepcopy(p);bad['source_impact_review']['fresh_worker_admission_required']=False
    assert u.gpu_group_order(bad,cp)==list(u.GROUPS)


def test_recorded_oom_requires_untampered_closed_job_and_seals(monkeypatch):
    from tools.user_acceptance import runner as u
    from tools.user_acceptance.recovery_review import verify_oom_cleanup
    folder=u.ROOT/'attempts/a1a33631f21c4f4383c64733ba806650'
    if not folder.exists():pytest.skip('local failure evidence not distributed')
    baseline=u.engine.read_json(u.ROOT/'benchmark-memory-baseline.json')['plan']
    verify_oom_cleanup(folder,baseline)
    original=u.engine.read_json
    for changes in ({'cleanup_verified':False},{'source_integrity':False},{'cleanup':{'active_processes':1}},{'cleanup_error':'unknown'}):
        def read(path,*a,changes=changes,**kw):
            value=original(path,*a,**kw)
            return dict(value,**changes) if Path(path).name=='controller.json' else value
        with monkeypatch.context() as m:
            m.setattr(u.engine,'read_json',read)
            with pytest.raises(RuntimeError,match='OOM_RECOVERY_UNVERIFIED'):verify_oom_cleanup(folder,baseline)


def test_u02_admission_includes_validation_scratch():
    import ast
    from tools.user_acceptance import runner as u
    tree=ast.parse((u.REPO/'tools/user_acceptance/scenarios.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='U02')
    assignment=next(n for n in ast.walk(fn) if isinstance(n,ast.Assign) and isinstance(n.value,ast.BinOp) and any(isinstance(t,ast.Name) and t.id=='gpu_working' for t in n.targets))
    assert 'validation_scratch' in ast.unparse(assignment.value)
