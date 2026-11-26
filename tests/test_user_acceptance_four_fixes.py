"""Focused CPU regressions; real CUDA validation belongs to explicit revalidation."""
import copy
import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from hyso import GraphModelBuilder,ModelConfigError,evaluate_classification
from hyso.models.output import GraphTensorOutput
from hyso.models.graph_ops import Fusion as Add
from hyso.evaluation import EvaluationResult
from hyso.performance.runtime_result import _hash
from tools.user_acceptance.cpu_onnx import canonical_trace_hash,verify_cpu_execution


def residual():
    return (GraphModelBuilder({'image':[1,4,4]})
        .add('conv','conv2d',['image'],in_channels=1,out_channels=1,kernel_size=1,padding=0)
        .add('skip','add',['image','conv'])
        .add('head','classification_head',['skip'],in_features=1,num_classes=2).outputs(logits='head').build())


def test_native_autocast_residual_forward_backward_update():
    model=GraphTensorOutput(residual(),'logits',task='multiclass')
    opt=torch.optim.SGD(model.parameters(),lr=.03)
    before=[p.detach().clone() for p in model.parameters()];observations={};handles=[]
    for name,node in model.graph.nodes.items():
        def capture(module,args,out,name=name):observations[name]=dict(shape=list(out.shape),dtype=str(out.dtype),device=str(out.device))
        handles.append(node.register_forward_hook(capture))
    try:
        with torch.autocast('cpu',dtype=torch.bfloat16):
            out=model(torch.ones(2,1,4,4));loss=nn.functional.cross_entropy(out,torch.tensor([0,1]))
        loss.backward();opt.step()
    finally:
        for handle in handles:handle.remove()
    assert observations['node_conv']['dtype']=='torch.bfloat16'
    assert observations['node_skip']['dtype']=='torch.float32'
    assert observations['node_head']['dtype']=='torch.bfloat16'
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(not torch.equal(a,b) for a,b in zip(before,model.parameters()))


@pytest.mark.parametrize('bad', ['dtype','shape','device'])
def test_autocast_still_rejects_invalid_node_contract(monkeypatch,bad):
    model=residual()
    def invalid(x):
        if bad=='dtype':return x.double()
        if bad=='shape':return x[:,:,:2,:2]
        return torch.empty_like(x,device='meta')
    monkeypatch.setattr(model.nodes['node_conv'],'forward',invalid)
    with torch.autocast('cpu',dtype=torch.bfloat16),pytest.raises(ModelConfigError) as error:
        model(torch.ones(2,1,4,4))
    assert error.value.code=='GRAPH_SHAPE_ERROR' and error.value.layer_id=='conv'
    assert all(k in error.value.actual for k in ('shape','dtype','device'))


def test_autocast_does_not_relax_inputs_or_non_autocast_outputs(monkeypatch):
    model=residual()
    with torch.autocast('cpu',dtype=torch.bfloat16),pytest.raises(ModelConfigError):
        model(torch.ones(2,1,4,4,dtype=torch.bfloat16))
    monkeypatch.setattr(model.nodes['node_conv'],'forward',lambda x:x.bfloat16())
    with pytest.raises(ModelConfigError):model(torch.ones(2,1,4,4))


def test_fusion_promotion_requires_active_policy_and_equal_shape_device():
    a=torch.ones(2,1,4,4);b=a.bfloat16();fusion=Add()
    with pytest.raises(ModelConfigError):fusion(a,b)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        assert fusion(a,b).dtype==torch.float32
        with pytest.raises(ModelConfigError):fusion(a,b.double())
        with pytest.raises(ModelConfigError):fusion(a,b[:,:,:2,:2])
        with pytest.raises(ModelConfigError):fusion(a,torch.empty_like(b,device='meta'))


def test_graph_trainer_gradient_diagnostics_real_update_integration():
    from hyso.training import Trainer
    from hyso.performance import measure_gradient_diagnostics
    model=GraphTensorOutput(residual(),'logits',task='multiclass')
    before=copy.deepcopy(model.state_dict())
    optimizer=torch.optim.SGD(model.parameters(),lr=.03)
    trainer=Trainer(model,optimizer=optimizer,loss_fn=nn.CrossEntropyLoss(),device='cpu',amp=False)
    data=DataLoader(TensorDataset(torch.ones(4,1,4,4),torch.tensor([0,1,0,1])),batch_size=2)
    result=measure_gradient_diagnostics(trainer,data,epochs=1).to_dict()
    assert result['summary']['applied']==2 and result['summary']['skipped']==0
    assert result['source_p19']['provenance']['runtime']['device']=='cpu'
    assert result['source_p19']['provenance']['training']['amp'] is False
    assert any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items())


@pytest.mark.parametrize('structure',[False,True])
def test_producer_and_structure_capture_preserve_model_and_rng(structure):
    model=nn.Sequential(nn.Flatten(),nn.Linear(16,2),nn.Dropout(.2));model.train();model[1].eval()
    for p in model.parameters():p.grad=torch.ones_like(p)
    state=copy.deepcopy(model.state_dict());modes=[m.training for m in model.modules()]
    gradients=[(p.grad,p.grad.clone()) for p in model.parameters()]
    rng=torch.get_rng_state().clone();py=random.getstate();nr=np.random.get_state()
    generator=torch.Generator().manual_seed(25)
    loader=DataLoader(TensorDataset(torch.arange(64,dtype=torch.float32).reshape(4,1,4,4)/64,torch.tensor([0,1,0,1])),
        batch_size=2,generator=generator)
    gr=generator.get_state().clone()
    result=evaluate_classification(model,loader,class_names=['a','b'],split='test',capture_evidence=True,capture_structure=structure)
    d=result.to_dict();assert d['schema_version']==(3 if structure else 2)
    assert d['producer_evidence']['evaluated_stream']['sample_count']==4 and d['source']['device']=='cpu'
    assert all(torch.equal(state[k],v) for k,v in model.state_dict().items()) and modes==[m.training for m in model.modules()]
    assert all(p.grad is original and torch.equal(p.grad,value) for p,(original,value) in zip(model.parameters(),gradients))
    assert torch.equal(rng,torch.get_rng_state()) and random.getstate()==py and torch.equal(gr,generator.get_state())
    after=np.random.get_state();assert nr[0]==after[0] and np.array_equal(nr[1],after[1]) and nr[2:]==after[2:]
    if structure:
        bad=copy.deepcopy(d);bad['source']['device']='cuda:0'
        bad['source_sha256']=_hash({k:v for k,v in bad.items() if k not in ('source_sha256','metrics','status')})
        with pytest.raises(ValueError):EvaluationResult(bad)


def test_trace_hash_exact_product_encoding_and_corruption():
    events=[{'name':'node Ω','args':{'provider':'CPUExecutionProvider'},'dur':.25}]
    assert canonical_trace_hash(events)==_hash(events)
    original=canonical_trace_hash(events);events[0]['dur']=.5
    assert canonical_trace_hash(events)!=original
    with pytest.raises(ValueError):canonical_trace_hash('trace.json')
    with pytest.raises(ValueError):canonical_trace_hash([{'dur':float('nan')}])


@pytest.mark.parametrize('flags',[{'capture_evidence':'yes'},{'capture_structure':True}])
def test_invalid_capture_flags_rejected_before_gpu_query(monkeypatch,flags):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('GPU queried before input validation'))
    with pytest.raises(ValueError):
        evaluate_classification(nn.Linear(2,2),None,class_names=['a','b'],split='test',device='cuda:0',**flags)


def test_recorded_cpu_onnx_positive_and_missing_run_negative():
    root=Path(__file__).resolve().parents[1]/'outputs/final-user-acceptance-20261126/attempts/16a7de52eba64a64b8ee383ba85fd0fe/U13'
    if not root.exists():pytest.skip('local sealed acceptance evidence is not distributed')
    result=json.loads((root/'onnx-provider-raw.json').read_text(encoding='utf-8'))
    identity=json.loads((root/'onnx-runtime-identity-raw.json').read_text(encoding='utf-8'))
    verify_cpu_execution(result,identity['loaded_runtime'],identity['planned_runtime'])
    assert result['evidence']['report']['completed_runs']==4 and len(result['analysis']['events'])==24
    bad=copy.deepcopy(result);bad['evidence']['report']['trace'][0]['dur']=12345
    with pytest.raises(ValueError):verify_cpu_execution(bad,identity['loaded_runtime'],identity['planned_runtime'])
    bad=copy.deepcopy(result);bad['analysis']['evidence_assessment']['node_coverage']['per_run_source_event_counts'][0].pop('/Add')
    with pytest.raises(ValueError):verify_cpu_execution(bad,identity['loaded_runtime'],identity['planned_runtime'])


def test_meta_activation_estimate_allocates_no_model_weights():
    from tools.user_acceptance.scenarios import meta_activation_bytes
    with torch.device('meta'):model=nn.Sequential(nn.Conv2d(3,4,1),nn.ReLU())
    value=meta_activation_bytes(model)
    assert value==3*4*32*32*4
    assert all(p.device.type=='meta' for p in model.parameters())
    assert not any(m._forward_hooks for m in model.modules())


def test_vgg11_real_meta_storage_and_corrected_gpu_budget():
    from torchvision.models import vgg11
    from tools.user_acceptance.scenarios import meta_activation_bytes
    with torch.device('meta'):model=vgg11(weights=None)
    parameters=sum(p.numel() for p in model.parameters())
    storage=sum(t.numel()*t.element_size() for t in [*model.parameters(),*model.buffers()])
    assert parameters==132863336 and storage==531453344
    activations=meta_activation_bytes(model)
    assert storage*4>676402298
    assert storage+3*32*32*4+activations<676402298
    assert storage*4<3*1024**3
