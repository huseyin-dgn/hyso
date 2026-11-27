"""E01-E06 focused real small CPU conversions and offline negative controls."""
import copy
from dataclasses import replace
import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
import hyso
from hyso import export as ex
from hyso.export._model import equivalence
from hyso.export.artifacts import Transaction,verify_files,file_hash,safe_file

class Small(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(3,2);self.register_buffer('bias',torch.ones(2))
    def forward(self,x):return self.linear(x)+self.bias

class Pair(torch.nn.Module):
    def forward(self,a,b):return a+b,a-b

class BadCopy(Small):
    def __deepcopy__(self,memo):return self

class Unsupported(torch.nn.Module):
    def forward(self,x):return torch.linalg.eigvals(x).real

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    assert torch.version.cuda is None
    torch.set_num_threads(1)
    def forbidden(*a,**k):raise AssertionError('export regression attempted CUDA execution')
    for n in ('_lazy_init','init','synchronize'):monkeypatch.setattr(torch.cuda,n,forbidden)

def state(m,x):
    return dict(weights=[(n,t.clone(),t.data_ptr(),t._version,t.stride()) for n,t in list(m.named_parameters())+list(m.named_buffers())],
        gradients=[(p,p.grad,None if p.grad is None else p.grad.clone()) for p in m.parameters()],modes=[(v,v.training) for v in m.modules()],
        cpu_rng=torch.get_rng_state().clone(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),input=x.clone())

def unchanged(before,m,x):
    for name,t,pointer,version,stride in before['weights']:
        now=dict(list(m.named_parameters())+list(m.named_buffers()))[name]
        assert torch.equal(now,t) and now.data_ptr()==pointer and now._version==version and now.stride()==stride
    for p,g,value in before['gradients']:assert p.grad is g and (g is None or torch.equal(g,value))
    assert all(v.training==mode for v,mode in before['modes'])
    assert torch.equal(torch.get_rng_state(),before['cpu_rng']) and random.getstate()==before['python_rng']
    assert np.array_equal(np.random.get_state()[1],before['numpy_rng'][1]) and torch.equal(x,before['input'])

@pytest.mark.parametrize('converter,suffix',[(ex.onnx,'onnx'),(ex.libtorch,'pt'),(ex.torch_export,'pt2')])
def test_real_conversion_runtime_manifest_reload_and_full_source_preservation(converter,suffix,tmp_path):
    model=Small().train();model.linear.eval();x=torch.randn(2,3)
    for p in model.parameters():p.grad=torch.ones_like(p)
    before=state(model,x);result=converter(model,x,tmp_path/('model.'+suffix));result.verify()
    d=result.to_dict();assert d['deployment_verified'] and all(v['status']=='PASS' for v in d['stages'].values())
    assert d['source_model']['weights_sha256'] and d['inputs'][0]['shape']==[2,3]
    assert ex.ExportResult.from_json(str(tmp_path/('model.'+suffix))+'.export.json').to_dict()==d
    assert not list(tmp_path.glob('.hyso-export-*'));unchanged(before,model,x)
    if converter is ex.onnx:
        assert d['runtime']['analysis']['events'] and all(e['provider']=='CPUExecutionProvider' for e in d['runtime']['analysis']['events'])
        assert d['runtime']['loaded_binaries'] and d['runtime']['output_trace_binding']

def test_script_is_real_and_cpp_status_separate(tmp_path):
    d=ex.libtorch(Small(),torch.ones(2,3),tmp_path/'script.pt',method='script').to_dict()
    assert d['deployment_verified'] and d['cpp_runtime']['status']=='NOT_RUN'

@pytest.mark.parametrize('converter,suffix',[(ex.onnx,'onnx'),(ex.libtorch,'pt'),(ex.torch_export,'pt2')])
def test_multi_input_multi_output_real(converter,suffix,tmp_path):
    d=converter(Pair(),(torch.ones(2,3),torch.ones(2,3)*2),tmp_path/('pair.'+suffix)).to_dict()
    assert len(d['inputs'])==len(d['outputs'])==2 and d['deployment_verified']

@pytest.mark.parametrize('converter,suffix',[(ex.onnx,'onnx'),(ex.libtorch,'pt')])
def test_real_modelbuilder_and_graphbuilder(converter,suffix,tmp_path):
    seq=hyso.ModelBuilder((1,4,4),2).conv2d(in_channels=1,out_channels=2,kernel_size=1).build()
    graph=hyso.GraphModelBuilder({'image':[1,4,4]}).add('conv','conv2d',['image'],in_channels=1,out_channels=2,kernel_size=1).outputs(features='conv').build()
    for name,model in [('builder',seq),('graph',graph)]:
        d=converter(model,torch.ones(2,1,4,4),tmp_path/(name+'.'+suffix)).to_dict()
        assert d['deployment_verified']

def test_dynamic_batch_onnx_and_exported_program_real_constraints(tmp_path):
    dyn={'image':{0:dict(name='batch',min=1,max=8)}};x=torch.ones(2,3)
    a=ex.onnx(Small(),x,tmp_path/'dynamic.onnx',input_names=['image'],dynamic_shapes=dyn)
    assert a.to_dict()['options']['dynamic_shapes']['image']['0']['max']==8
    import onnxruntime as ort
    session=ort.InferenceSession(str(tmp_path/'dynamic.onnx'),providers=['CPUExecutionProvider'])
    assert session.run(None,{'image':np.ones((4,3),np.float32)})[0].shape== (4,2)
    ex.torch_export(Small(),x,tmp_path/'dynamic.pt2',input_names=['image'],dynamic_shapes=dyn)
    program=torch.export.load(tmp_path/'dynamic.pt2').module()
    assert program(torch.ones(4,3)).shape==(4,2)
    with pytest.raises(RuntimeError):program(torch.ones(9,3))

@pytest.mark.parametrize('x',[torch.tensor([[float('nan')]*3]),torch.ones(2,3,dtype=torch.int64),torch.ones(2,3,requires_grad=True)])
def test_invalid_input_rejected_without_artifact(x,tmp_path):
    with pytest.raises((ex.ExportError,ValueError)):ex.libtorch(Small(),x,tmp_path/'bad.pt')
    assert list(tmp_path.iterdir())==[]

def test_failed_export_preserves_model_rng_and_user_files(tmp_path):
    model=Unsupported();x=torch.eye(3);before=state(model,x)
    keep=tmp_path/'keep';keep.write_text('user')
    with pytest.raises(ex.ExportError) as error:ex.onnx(model,x,tmp_path/'bad.onnx')
    assert error.value.result['stages']['conversion']['status']=='FAIL'
    unchanged(before,model,x);assert list(tmp_path.iterdir())==[keep]

def test_copy_alias_rejected(tmp_path):
    with pytest.raises(ex.ExportError,match='MODEL_COPY'):ex.libtorch(BadCopy(),torch.ones(2,3),tmp_path/'bad.pt')

@pytest.mark.parametrize('field,value',[('atol',float('nan')),('rtol',-1),('max_model_bytes',2**40),('overwrite',1)])
def test_invalid_budgets_and_tolerances(field,value):
    with pytest.raises(ex.ExportError):replace(ex.ExportConfig(),**{field:value})

def test_model_and_input_budget_rejected_before_forward(tmp_path,monkeypatch):
    m=Small();monkeypatch.setattr(m,'forward',lambda *a:pytest.fail('budget guard ran forward'))
    with pytest.raises(ex.ExportError,match='RESOURCE_BUDGET'):ex.libtorch(m,torch.ones(2,3),tmp_path/'small.pt',config=ex.ExportConfig(max_model_bytes=1))
    assert list(tmp_path.iterdir())==[]

@pytest.mark.parametrize('actual',[torch.zeros(2,3,dtype=torch.float64),torch.zeros(3,2),torch.full((2,3),float('inf')),torch.ones(2,3)])
def test_output_mismatch_dtype_shape_finite_and_tolerance(actual):
    with pytest.raises(ex.ExportError):equivalence(torch.zeros(2,3),actual,ex.ExportConfig())

def test_noncontiguous_equivalence_and_chunk_boundary_nonfinite():
    a=torch.arange(131072,dtype=torch.float32).reshape(256,512).t()
    assert equivalence(a,a.contiguous(),ex.ExportConfig())[0]['matched']
    bad=a.clone();bad[128,128]=float('nan')
    with pytest.raises(ex.ExportError,match='NONFINITE'):equivalence(a,bad,ex.ExportConfig())

def test_no_clobber_explicit_overwrite_and_failed_overwrite_rollback(tmp_path):
    p=tmp_path/'model.pt';m=Small();x=torch.ones(2,3)
    first=ex.libtorch(m,x,p);previous=p.read_bytes();side=Path(str(p)+'.export.json').read_bytes()
    with pytest.raises(ex.ExportError,match='EXISTS'):ex.libtorch(m,x,p)
    assert p.read_bytes()==previous
    ex.libtorch(m,x,p,config=ex.ExportConfig(overwrite=True)).verify()
    latest=p.read_bytes();metadata=Path(str(p)+'.export.json').read_bytes()
    with pytest.raises(ex.ExportError):ex.onnx(Unsupported(),torch.eye(3),p,config=ex.ExportConfig(overwrite=True))
    assert p.read_bytes()==latest and Path(str(p)+'.export.json').read_bytes()==metadata

def test_artifact_and_metadata_tampering_rejected(tmp_path):
    p=tmp_path/'model.pt';r=ex.libtorch(Small(),torch.ones(2,3),p)
    d=r.to_dict();d['backend']='forged'
    with pytest.raises(ex.ExportError,match='RESULT_INTEGRITY'):ex.ExportResult(d)
    p.write_bytes(p.read_bytes()+b'forged')
    with pytest.raises(ex.ExportError,match='ARTIFACT_INTEGRITY'):r.verify()
    with pytest.raises(ex.ExportError):ex.ExportResult.from_json(str(p)+'.export.json')

def test_multi_file_manifest_and_partial_publication_rollback(tmp_path,monkeypatch):
    from hyso.export.contracts import seal,STAGES
    import os
    output=tmp_path/'main';tx=Transaction(output,False,1024)
    with tx:
        (tx.root/'main').write_bytes(b'main');(tx.root/'external.data').write_bytes(b'external')
        body=dict(schema_version=1,format='hyso.export',artifact_root=str(tmp_path),artifact_path=str(output),artifact_files=tx.inventory(),stages={n:dict(status='NOT_RUN') for n in STAGES})
        result=seal(body);real=os.link;count=[0]
        def fail_second(a,b):
            count[0]+=1
            if count[0]==2:raise OSError('simulated partial publication')
            return real(a,b)
        monkeypatch.setattr(os,'link',fail_second)
        with pytest.raises(OSError):tx.publish(result)
    assert list(tmp_path.iterdir())==[]

@pytest.mark.parametrize('name',['../escape','/absolute','C:/escape','a\\escape'])
def test_manifest_path_escapes_rejected(name,tmp_path):
    with pytest.raises(ex.ExportError):safe_file(tmp_path,name)

def test_missing_ownership_marker_refuses_cleanup(tmp_path):
    tx=Transaction(tmp_path/'model',False,1024);(tx.root/'.owned.json').unlink()
    with pytest.raises((ex.ExportError,FileNotFoundError)):tx.__exit__(None,None,None)
    assert tx.root.exists()

def test_missing_dependency_blocks_before_intermediate_or_native_work(tmp_path,monkeypatch):
    from hyso.export import native
    original=native.importlib.util.find_spec
    monkeypatch.setattr(native.importlib.util,'find_spec',lambda name:None if name=='tensorrt' else original(name))
    for f,args in [(ex.tensorrt,('missing.onnx',tmp_path/'a.engine')),(ex.pytorch_tensorrt,(Small(),torch.ones(2,3),tmp_path/'b.engine'))]:
        with pytest.raises(ex.ExportError) as e:f(*args)
        assert e.value.code=='MISSING_DEPENDENCY' and e.value.result['stages']['dependencies']['status']=='BLOCKED'
    assert list(tmp_path.iterdir())==[]

@pytest.mark.parametrize('profile',[{'x':{'min':[4,3],'opt':[2,3],'max':[8,3]}},{'unknown':{}},{'x':{'min':[1,3],'opt':[2,4],'max':[8,4]}}])
def test_tensorrt_profile_bounds_negative(profile):
    from hyso.export.native import validate_profiles
    with pytest.raises(ex.ExportError):validate_profiles(profile,{'x':[-1,3]})

def test_no_runtime_request_is_not_deployment_pass(tmp_path):
    d=ex.libtorch(Small(),torch.ones(2,3),tmp_path/'unverified.pt',config=ex.ExportConfig(validate_runtime=False)).to_dict()
    assert not d['deployment_verified'] and d['stages']['runtime_execution']['status']=='NOT_RUN'

def test_aot_windows_is_blocked_without_compiler_launch(tmp_path):
    ep=ex.torch_export(Small(),torch.ones(2,3),tmp_path/'program.pt2')
    with pytest.raises(ex.ExportError) as e:ex.aotinductor(ep,tmp_path/'compiled')
    assert e.value.code=='AOT_PLATFORM' and e.value.result['stages']['admission']['status']=='BLOCKED'

@pytest.mark.parametrize('dynamic',[False,True])
def test_real_modern_onnx_exporter_strategy_and_optional_dynamic_batch(dynamic,tmp_path):
    kwargs=dict(input_names=['image'],backend='dynamo',opset_version=18)
    if dynamic:kwargs['dynamic_shapes']={'image':{0:dict(name='batch',min=1,max=8)}}
    result=ex.onnx(torch.nn.Linear(3,2),torch.ones(2,3),tmp_path/'dynamo.onnx',**kwargs)
    d=result.to_dict()
    assert d['deployment_verified'] and d['options']['actual_capture_strategy'].startswith('torch.export.export(')
    assert d['options']['legacy_fallback_allowed'] is False and any(r['path'].endswith('.md') for r in d['artifact_files'])
    if dynamic:
        import onnxruntime as ort
        session=ort.InferenceSession(d['artifact_path'],providers=['CPUExecutionProvider'])
        assert session.run(None,{'image':np.ones((4,3),np.float32)})[0].shape==(4,2)

def test_real_external_data_shared_file_is_manifest_bound(tmp_path,monkeypatch):
    import onnx
    exporter=torch.onnx.export
    def external_export(*args,**kwargs):
        exporter(*args,**kwargs)
        path=Path(args[2]);loaded=onnx.load(str(path))
        onnx.save_model(loaded,str(path),save_as_external_data=True,all_tensors_to_one_file=True,location='shared.data',size_threshold=0)
    monkeypatch.setattr(torch.onnx,'export',external_export)
    r=ex.onnx(Small(),torch.ones(2,3),tmp_path/'external.onnx');d=r.to_dict();r.verify()
    files=[v for v in d['artifact_files'] if '.data.' in v['path']]
    assert len(files)==1 and files[0]['path'].startswith('external.onnx.data.')
    path=tmp_path/files[0]['path'];path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ex.ExportError,match='ARTIFACT_INTEGRITY'):r.verify()

def test_rehashed_fabricated_error_summary_is_rejected(tmp_path):
    from hyso.export.contracts import seal
    from tools.export_acceptance.evidence import verify_comparisons
    r=ex.libtorch(Small(),torch.ones(2,3),tmp_path/'actual.pt');d=r.to_dict()
    verify_comparisons(d)
    d['stages']['equivalence']['comparisons'][0]['max_absolute_error']=10
    with pytest.raises(ex.ExportError,match='RESULT_INTEGRITY'):seal(d)

def test_result_is_detached_and_immutable(tmp_path):
    r=ex.libtorch(Small(),torch.ones(2,3),tmp_path/'actual.pt');d=r.to_dict();d['backend']='changed'
    assert r.to_dict()['backend']!='changed'
    with pytest.raises(AttributeError):r._payload='forged'

def test_retained_p35_cpu_and_cuda_parser_compatibility_without_execution():
    from hyso.performance import RuntimeResult
    from hyso.performance.providers import _analyze_providers
    from hyso.performance.runtime_result import _hash
    root=Path(__file__).resolve().parents[1]/'outputs/final-user-acceptance-20261126'
    cp=json.loads((root/'checkpoint.json').read_text())
    cpu=Path(cp['cpu_results']['tests/test_user_acceptance_scenarios.py::test_U13']['evidence']).parent/'U13/onnx-provider-raw.json'
    gpu=Path(cp['results']['U13']['evidence'])
    before={p:file_hash(p) for p in (cpu,gpu)}
    rows=[json.loads(cpu.read_text()),json.loads(gpu.read_text())['data']['observations']['result']]
    for row in rows:
        restored=RuntimeResult(row).to_dict();assert restored==row
        e=copy.deepcopy(row['evidence']);e['report']['trace']=[];e['report']['trace_sha256']=_hash([])
        e['report']['profile_binding']=_hash(dict(session_id=e['token'],model_sha256=e['report']['model_sha256'],trace_sha256=e['report']['trace_sha256'],input_sha256=e['report']['input_sha256'],providers=e['report']['session_providers']))
        assert _analyze_providers(e)['status']!='expected_execution_observed'
    assert {p:file_hash(p) for p in (cpu,gpu)}==before

def test_export_public_hints_guide_and_commercial_profile():
    import inspect,typing,hyso_guide
    assert 'export' in hyso.__all__ and hyso.export is ex
    for name in ex.__all__:
        value=getattr(ex,name)
        if inspect.isfunction(value):assert typing.get_type_hints(value)['return'] is not None
        assert hyso_guide.topic_for_api('hyso.export.'+name)=='export.overview'
    for lang in ('tr','en','de','es'):
        text=hyso_guide.show_topic('export.overview',lang)['text']
        assert 'E01' in text and 'E06' in text and 'full_cuda_acceptance=false' in text
    root=Path(__file__).resolve().parents[1];profile=json.loads((root/'commercial-profile.json').read_text())
    assert {p.relative_to(root).as_posix() for p in (root/'hyso/export').glob('*.py')}<=set(profile['python_files'])

@pytest.mark.parametrize('damage',['shape','dtype'])
def test_valid_but_source_incompatible_onnx_rejected_before_runtime(tmp_path,monkeypatch,damage):
    import onnx
    from hyso.export import runtime
    def incompatible(model,args,path,**kwargs):
        dtype=onnx.TensorProto.DOUBLE if damage=='dtype' else onnx.TensorProto.FLOAT
        graph=onnx.helper.make_graph([onnx.helper.make_node('Identity',['input0'],['output0'],name='identity')],'incompatible',
            [onnx.helper.make_tensor_value_info('input0',dtype,[2,3])],[onnx.helper.make_tensor_value_info('output0',dtype,[2,3])])
        onnx.save(onnx.helper.make_model(graph,opset_imports=[onnx.helper.make_opsetid('',17)]),path)
    monkeypatch.setattr(torch.onnx,'export',incompatible)
    monkeypatch.setattr(runtime,'ort_cpu',lambda *a:pytest.fail('incompatible artifact reached runtime'))
    with pytest.raises(ex.ExportError,match='IO_CONTRACT') as e:ex.onnx(Small(),torch.ones(2,3),tmp_path/'bad.onnx')
    assert e.value.result['stages']['artifact']['status']=='FAIL' and not list(tmp_path.iterdir())

def test_cleanup_failure_is_ownership_failure_and_keeps_evidence(tmp_path,monkeypatch):
    from hyso.export import artifacts
    tx=Transaction(tmp_path/'result.pt',False,1024)
    def denied(*args,**kwargs):raise PermissionError('simulated locked owned artifact')
    monkeypatch.setattr(artifacts.shutil,'rmtree',denied)
    with pytest.raises(ex.ExportError,match='OWNERSHIP'):tx.__exit__(None,None,None)
    assert tx.root.exists() and (tx.root/'.owned.json').exists()

def test_runtime_artifact_binding_cannot_be_rehashed_to_another_model(tmp_path):
    from hyso.export.contracts import seal
    d=ex.onnx(Small(),torch.ones(2,3),tmp_path/'bound.onnx').to_dict()
    assert d['runtime']['artifact_binding']['main_sha256']==d['artifact_sha256']
    d['runtime']['artifact_binding']['main_sha256']='0'*64
    with pytest.raises(ex.ExportError,match='RESULT_INTEGRITY'):seal(d)
