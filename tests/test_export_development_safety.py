"""New export negative/provenance/CLI checks; GPU allocator is simulated only."""
import base64
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from hyso import export as ex
from hyso.export import dependencies


def installed(tmp_path):
    python=tmp_path/'python.exe';python.write_bytes(b'fake interpreter; never executed')
    site=tmp_path/'Lib/site-packages';site.mkdir(parents=True)
    for name in ('tensorrt-cu12','tensorrt-cu12-libs','tensorrt-cu12-bindings'):
        info=site/(name.replace('-','_')+'-10.13.3.9.dist-info');info.mkdir()
        (info/'METADATA').write_text('Metadata-Version: 2.1\nName: '+name+'\nVersion: 10.13.3.9\n')
        member=name+'.dll';file=site/member;file.write_bytes(name.encode())
        with (info/'RECORD').open('w',newline='') as stream:
            csv.writer(stream).writerow([member,'sha256='+base64.urlsafe_b64encode(hashlib.sha256(file.read_bytes()).digest()).decode().rstrip('='),file.stat().st_size])
    return python,site


@pytest.mark.parametrize('damage',[None,'hash','size','outside','numpy_remnant'])
def test_record_backed_trt_inventory(tmp_path,damage):
    python,site=installed(tmp_path)
    if damage=='hash':
        p=site/'tensorrt-cu12-libs.dll';p.write_bytes(b'X'*p.stat().st_size)
    if damage=='size':(site/'tensorrt-cu12-libs.dll').write_bytes(b'x')
    if damage=='outside':
        info=next(site.glob('tensorrt_cu12_libs-*.dist-info'))
        (info/'RECORD').write_text('../../foreign.dll,,\n')
        (tmp_path/'foreign.dll').write_bytes(b'foreign')
    if damage=='numpy_remnant':(site/'~umpy-2.2.6.dist-info').mkdir()
    if damage in ('hash','size','outside'):
        with pytest.raises((ValueError,OSError)):dependencies.runtime_inventory(python)
    else:
        result=dependencies.runtime_inventory(python)
        assert result['tensorrt']['status']==('BLOCKED' if damage else 'READY_FOR_ADMISSION')
        assert len(result['tensorrt']['provenance'])==3 and not result['tensorrt']['runtime_verified']


def test_allocator_callbacks_offline():
    from hyso.export._worker import gpu_allocator
    events=[]
    class Buffer:
        def __init__(self,n):self.n=n
        def data_ptr(self):return 4096
        def numel(self):return self.n
    fake=SimpleNamespace(uint8='uint8',OutOfMemoryError=MemoryError,empty=lambda shape,**kw:Buffer(shape[0]),
                         cuda=SimpleNamespace(synchronize=lambda device:events.append('sync')))
    allocator=gpu_allocator(SimpleNamespace(IGpuAllocator=object),fake,32)
    assert allocator.allocate(0,0,0)==0
    pointer=allocator.allocate_async(16,256,0,77)
    assert pointer==4096 and allocator.used==16
    assert allocator.reallocate(pointer,256,24)==0 and allocator.used==16
    assert allocator.allocate(20,256,0)==0 and allocator.denied
    assert allocator.deallocate_async(pointer,77) and allocator.used==0
    assert allocator.deallocate(0) and not allocator.deallocate(999)
    assert events==['sync','sync']


def test_optional_imports_missing(monkeypatch):
    import importlib.util
    original=importlib.util.find_spec
    monkeypatch.setattr(importlib.util,'find_spec',lambda n,*a,**k:None if n in ('tensorrt','onnxruntime','onnxscript') else original(n,*a,**k))
    available=ex.availability()
    assert not available['dependencies']['tensorrt']['discoverable']
    with pytest.raises(ex.ExportError,match='MISSING_DEPENDENCY'):ex.tensorrt('missing.onnx','unused.engine')


def test_saved_record_and_report_integrity(tmp_path):
    import numpy as np
    row=ex.output_metrics(np.ones(2),np.ones(2));p=tmp_path/'result.json'
    ex.save(row,p);assert ex.load(p)==row
    with pytest.raises(Exception):ex.save(row,p)
    changed=json.loads(p.read_text(encoding='utf-8'));changed['evidence']['numerical']['status']='FAIL'
    p.write_text(json.dumps(changed),encoding='utf-8')
    with pytest.raises(ex.ExportError,match='RESULT_INTEGRITY'):ex.load(p)


def test_cli_saved_report_and_execution_gate(tmp_path):
    import numpy as np
    from hyso.cli.main import build_parser
    parser=build_parser();p=tmp_path/'metrics.json';ex.save(ex.output_metrics(np.ones(2),np.ones(2)),p)
    args=parser.parse_args(['export','report',str(p),'--language','de','--format','html'])
    result=args.handler(args);assert 'Deployment-Exportbericht' in result.text
    args=parser.parse_args(['export','benchmark','missing.json','--example','missing.npz'])
    with pytest.raises(ValueError,match='--execute'):args.handler(args)


def test_cross_runtime_named_inputs_and_rng(tmp_path):
    class Identity(torch.nn.Module):
        def forward(self,x):return x+1
    x=torch.ones(2,3);a=ex.libtorch(Identity(),x,tmp_path/'model.pt')
    before=torch.random.get_rng_state().clone()
    result=ex.Runtime(a).run({'input0':x})
    assert torch.equal(result[0],x+1) and torch.equal(before,torch.random.get_rng_state())


def test_native_benchmark_never_uses_resident_session(monkeypatch):
    runtime=object.__new__(ex.Runtime)
    runtime.device='cuda';runtime.info={'backend':'tensorrt_engine'}
    monkeypatch.setattr(runtime,'measure_native',lambda *a,**k:(_ for _ in ()).throw(ex.ExportError('CUDA_ADMISSION_REQUIRED','test gate')))
    with pytest.raises(ex.ExportError,match='CUDA_ADMISSION_REQUIRED'):ex.benchmark(runtime,torch.ones(2,3),iterations=1)


def test_e13_prepare_keeps_historical_evidence(tmp_path):
    from tools.export_acceptance.development import prepare
    (tmp_path/'selected').mkdir()
    python,_=installed(tmp_path/'selected')
    path=prepare(python,tmp_path/'e13')
    result=json.loads(path.read_text(encoding='utf-8'))
    assert result['prepare_only'] and not result['gpu_execution_started']
    assert result['runtime']['tensorrt']['status']=='READY_FOR_ADMISSION'
    assert result['preserved_evidence'] and not result['evidence_impact']['historical_records_modified']
    assert len(result['gpu_scenarios'])==7


def test_selected_runtime_request_uses_separate_plan(monkeypatch,tmp_path):
    from tools.export_acceptance import runner
    original=dict(python='baseline',site='baseline')
    selected=dict(runtime=dict(python='selected',site='selected'),feature_tools={'tensorrt':{'status':'READY_FOR_ADMISSION'}},issues=[])
    plan=dict(runtime=original,native_runtime=selected)
    monkeypatch.setattr(runner,'BASE_REQUEST',lambda folder,work,group,plan,inventory:dict(gpu_plan=plan))
    request=runner.request_for(tmp_path,tmp_path,'03-native-engine',plan)
    assert request['gpu_plan']['runtime']['python']=='selected' and plan['runtime']==original
    selected['feature_tools']['tensorrt']['status']='BLOCKED'
    with pytest.raises(RuntimeError,match='TENSORRT_DEPENDENCIES_BLOCKED'):runner.request_for(tmp_path,tmp_path,'03-native-engine',plan)
