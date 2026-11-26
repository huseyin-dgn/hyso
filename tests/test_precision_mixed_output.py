"""CPU-only precision contract regressions; CUDA acceptance remains explicit."""
import copy
from contextlib import nullcontext
import pytest
import torch
from torch import nn
from hyso.quality import (EvaluationCohort,QualityExperimentResult,evaluate_precision_quality,
    _PrecisionAdapter,_hash)
from hyso.checkpoint import _rng_state
from test_quality_experiments import equal


class OffsetClassifier(nn.Module):
    def __init__(self):
        super().__init__();self.head=nn.Linear(4,2);self.register_buffer('offset',torch.zeros(1))
    def forward(self,x):return self.head(x.flatten(1))+self.offset


def cohort():
    return EvaluationCohort(torch.arange(32,dtype=torch.float32).reshape(8,1,2,2)/32,[0,1]*4,
        sample_ids=[str(i) for i in range(8)],class_names=['a','b'],split='holdout',preprocessing_id='local-arange-v1')


def test_real_mixed_output_has_actual_dtype_and_native_operator_evidence():
    c=cohort();model=OffsetClassifier()
    d=evaluate_precision_quality(model,c,candidate='bf16',batch_size=3).to_dict()
    a,b=d['sources']['baseline_precision'],d['sources']['candidate_precision']
    assert a['execution_version']==2 and b['execution_version']==3
    assert a['source_master_sha256']==b['source_master_sha256']==d['sources']['master_state_sha256']
    assert a['observed_input_stream_sha256']==b['observed_input_stream_sha256']
    assert d['sources']['baseline']['raw']['targets']==d['sources']['candidate']['raw']['targets']==c.to_dict()['targets']
    assert b['verified_execution']['native_output_dtype']=='torch.float32'
    assert b['verified_execution']['native_output_dtypes']==['torch.float32']*3
    for row in b['batch_observations']:
        assert row['native_output_dtype']=='torch.float32' and row['autocast_enabled'] is True
        assert row['autocast_dtype']=='torch.bfloat16' and row['operators']
        assert all(op['output_dtype']=='torch.bfloat16' and op['weight_dtype']=='torch.float32' for op in row['operators'])
    assert QualityExperimentResult(d).to_dict()==d


def test_fp16_output_promotion_cpu_proxy_not_cuda_acceptance():
    model=OffsetClassifier();observed=[];native_outputs=[]
    handle=model.head.register_forward_hook(lambda m,a,o:observed.append((str(a[0].dtype),str(m.weight.dtype),str(o.dtype))))
    native=model.register_forward_hook(lambda m,a,o:native_outputs.append(str(o.dtype)))
    try:out=_PrecisionAdapter(model,'amp_fp16')(torch.ones(4,1,2,2))
    finally:handle.remove();native.remove()
    assert observed==[('torch.float32','torch.float32','torch.float16')]
    assert out.dtype==torch.float32 and torch.promote_types(torch.float16,torch.float32)==torch.float32
    assert native_outputs==['torch.float32']


@pytest.mark.parametrize('damage',['summary','no_ops','op_dtype','op_context','batch_context','batch_dtype','input_dtype','device','weights','version','target'])
def test_rehashed_false_mixed_execution_proofs_rejected(damage):
    d=evaluate_precision_quality(OffsetClassifier(),cohort(),candidate='bf16',batch_size=4).to_dict()
    proof=d['sources']['candidate_precision'];row=proof['batch_observations'][0]
    if damage=='summary':proof['verified_execution']['native_output_dtype']='torch.bfloat16'
    elif damage=='no_ops':row['operators']=[]
    elif damage=='op_dtype':row['operators'][0]['output_dtype']='torch.float32'
    elif damage=='op_context':row['operators'][0]['autocast_enabled']=False
    elif damage=='batch_context':row['autocast_enabled']=False
    elif damage=='batch_dtype':row['native_output_dtype']='torch.float64'
    elif damage=='input_dtype':row['operators'][0]['input_dtype']='torch.float64'
    elif damage=='device':row['operators'][0]['device']='meta'
    elif damage=='weights':row['operators'][0]['weight_dtype']='torch.bfloat16'
    elif damage=='version':proof['execution_version']=99
    else:row['autocast_dtype']='torch.float16'
    d['source_sha256']=_hash(d['sources'])
    with pytest.raises(ValueError):QualityExperimentResult(d)


@pytest.mark.parametrize('output_dtype',[torch.float32,torch.bfloat16])
def test_fp32_or_forced_low_output_without_native_amp_operator_is_not_evidence(output_dtype):
    class NoEligible(nn.Module):
        def forward(self,x):return x.flatten(1)[:,:2].to(output_dtype)
    adapter=_PrecisionAdapter(NoEligible(),'bf16')
    error=ValueError if output_dtype==torch.float32 else NotImplementedError
    with pytest.raises(error,match='operator execution'):adapter(torch.ones(2,1,2,2))


def test_native_fp32_operator_fallback_cannot_be_hidden_by_output_cast():
    class DisabledLinear(nn.Linear):
        def forward(self,x):
            with torch.autocast(x.device.type,enabled=False):return nn.functional.linear(x.float(),self.weight,self.bias)
    class Fake(nn.Module):
        def __init__(self):super().__init__();self.head=DisabledLinear(4,2)
        def forward(self,x):return self.head(x.flatten(1)).bfloat16()
    model=Fake()
    with pytest.raises(ValueError,match='operator fallback'):_PrecisionAdapter(model,'bf16')(torch.ones(2,1,2,2))
    assert not model.head._forward_hooks


@pytest.mark.parametrize('mode',['disabled','wrong_target'])
def test_actual_autocast_policy_must_match(monkeypatch,mode):
    original=torch.autocast
    if mode=='disabled':monkeypatch.setattr(torch,'autocast',lambda *a,**kw:nullcontext())
    else:monkeypatch.setattr(torch,'autocast',lambda device,**kw:original(device,dtype=torch.float16,enabled=True))
    model=OffsetClassifier()
    with pytest.raises(ValueError,match='autocast policy'):_PrecisionAdapter(model,'bf16')(torch.ones(2,1,2,2))
    assert not model.head._forward_hooks


@pytest.mark.parametrize('bad',['dtype','device'])
def test_wrong_native_output_is_rejected(bad):
    class Invalid(OffsetClassifier):
        def forward(self,x):
            out=super().forward(x)
            return out.double() if bad=='dtype' else torch.empty_like(out,device='meta')
    with pytest.raises(ValueError,match='fallback|device'):_PrecisionAdapter(Invalid(),'bf16')(torch.ones(2,1,2,2))


def test_wrong_master_parameter_dtype_is_rejected():
    with pytest.raises((ValueError,RuntimeError)):_PrecisionAdapter(OffsetClassifier().double(),'bf16')(torch.ones(2,1,2,2))


def test_cuda_evidence_gate_schema_replay_v2_v3_and_false_amp_rejection():
    """Offline schema fixtures only; never emitted as a new GPU measurement."""
    import json
    from tools.user_acceptance import runner as u
    from tools.cuda_acceptance.feature_evidence import verify_feature
    from tools.cuda_acceptance.p25_evidence import native_hash
    from hyso.quality import _precision_execution_summary
    found=None
    for path in (u.CUDA_ROOT/'attempts').glob('*/P28.json'):
        artifact=json.loads(path.read_text(encoding='utf-8'))
        if artifact.get('status')=='PASS':found=(path,artifact['data']);break
    if found is None:pytest.skip('local retained CUDA evidence not distributed')
    path,original=found;runtime=json.loads((path.parent/'gpu-runtime.json').read_text(encoding='utf-8'))
    verify_feature('P28',original,runtime)
    mixed=copy.deepcopy(original);proof=mixed['sources']['candidate_precision'];proof['execution_version']=3
    for row in proof['batch_observations']:row['native_output_dtype']='torch.float32'
    proof['verified_execution']=_precision_execution_summary('amp_fp16',len(proof['batch_observations']),proof['batch_observations'])
    mixed['source_sha256']=native_hash(mixed['sources']);verify_feature('P28',mixed,runtime)
    for change in ('no_ops','fp32_operator','disabled','wrong_target','dtype'):
        bad=copy.deepcopy(mixed);row=bad['sources']['candidate_precision']['batch_observations'][0]
        if change=='no_ops':row['operators']=[]
        elif change=='fp32_operator':row['operators'][0]['output_dtype']='torch.float32'
        elif change=='disabled':row['autocast_enabled']=False
        elif change=='wrong_target':row['operators'][0]['autocast_dtype']='torch.bfloat16'
        else:row['native_output_dtype']='torch.float64'
        bad['source_sha256']=native_hash(bad['sources'])
        with pytest.raises(ValueError):verify_feature('P28',bad,runtime)


@pytest.mark.parametrize('fail',[False,True])
def test_master_cohort_modes_gradients_and_rng_preserved(fail):
    class Invalid(OffsetClassifier):
        def forward(self,x):return super().forward(x).double()
    model=Invalid() if fail else OffsetClassifier();model.train();model.head.eval()
    for p in model.parameters():p.grad=torch.ones_like(p)
    state=copy.deepcopy(model.state_dict());refs=[p.grad for p in model.parameters()]
    gradients=[p.grad.clone() for p in model.parameters()];modes=[m.training for m in model.modules()]
    c=cohort();data=c.to_dict();rng=_rng_state()
    if fail:
        with pytest.raises(ValueError):evaluate_precision_quality(model,c,candidate='bf16',batch_size=4)
    else:evaluate_precision_quality(model,c,candidate='bf16',batch_size=4)
    equal(state,model.state_dict());equal(rng,_rng_state())
    assert data==c.to_dict() and modes==[m.training for m in model.modules()]
    assert all(p.grad is ref and torch.equal(p.grad,value) for p,ref,value in zip(model.parameters(),refs,gradients))
