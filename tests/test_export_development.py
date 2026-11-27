"""E07-E12 small CPU engineering checks; no CUDA queries or native engines."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from hyso import export as ex


class Matrix(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor([[.1,.2],[.3,.4],[.5,.6]]))
    def forward(self,x):return x@self.weight


@pytest.fixture
def source(tmp_path):
    m=Matrix();x=torch.ones(2,3)
    return ex.onnx(m,x,tmp_path/'source.onnx'),x,m


def test_metrics_multioutput_dtype_and_nonfinite():
    a=np.array([1.,2.],dtype='float32');b=a.astype('float16')
    r=ex.output_metrics((a,a),(b,b),precision='fp16')
    assert r['numerical']['status']=='PASS' and not r['numerical']['outputs'][0]['dtype_match']
    assert ex.output_metrics(a,b)['numerical']['status']=='FAIL'
    assert ex.output_metrics(a,np.array([np.nan,2.]))['numerical']['status']=='FAIL'
    assert ex.output_metrics(a,np.ones(3))['numerical']['outputs'][0]['mae'] is None
    assert r['quality']['status']=='NOT_TESTED'


@pytest.mark.parametrize('precision',['fp32','fp16','int8'])
def test_tolerance_contract(precision):
    assert ex.Tolerance.for_precision(precision).atol>0
    with pytest.raises(ex.ExportError):ex.Tolerance(float('nan'),.1)


def test_quality_known_values():
    scores=np.array([[2.,1.],[.1,.9],[.8,.2]])
    row=ex.classification_metrics(scores,np.array([0,1,1]),top_k=2)
    assert row['top1_accuracy']==pytest.approx(2/3) and row['topk_accuracy']==1
    assert row['precision']==pytest.approx(.75)
    assert row['recall']==pytest.approx(.75)
    assert row['f1']==pytest.approx(2/3)
    with pytest.raises(ex.ExportError):ex.classification_metrics(scores,np.array([0,-1,1]))


def test_fp16_real_graph_and_source_preserved(source,tmp_path):
    a,x,m=source;before=copy.deepcopy(m.state_dict());rng=torch.random.get_rng_state().clone()
    out=ex.fp16(a,tmp_path/'half.onnx',example_input={'input0':x})
    row=out.to_dict();out.verify();a.verify()
    assert row['precision']['fp16_initializers']>0
    assert row['precision']['keep_io_types'] and not row['precision']['all_operations_in_requested_dtype']
    assert row['precision_validation']['numerical']['status']=='PASS'
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())


def test_dynamic_int8_real_artifact(source,tmp_path):
    a,x,_=source
    out=ex.quantize(a,tmp_path/'dynamic.onnx',example_input={'input0':x})
    row=out.to_dict();assert row['precision']['mode']=='dynamic'
    assert 'MatMulInteger' in row['precision']['observed_op_types']
    assert row['precision']['calibration']['status']=='not_required'
    assert row['precision_validation']['numerical']['status']=='PASS'
    assert ex.ExportResult.from_json(str(tmp_path/'dynamic.onnx')+'.export.json').to_dict()==row


def test_static_qdq_real_calibration(source,tmp_path):
    a,x,_=source
    out=ex.quantize(a,tmp_path/'static.onnx',mode='static',calibration=[{'input0':x},{'input0':x*.5}],
                    dataset_id='synthetic_test_only',representative=False,example_input={'input0':x})
    row=out.to_dict();cal=row['precision']['calibration']
    assert cal['batches']==2 and cal['content_sha256'] and cal['representative_declared'] is False
    assert row['precision']['format']=='QDQ' and row['precision']['native_validation']=='NOT_TESTED'


@pytest.mark.parametrize('calibration', [[],[{'input0':np.ones((2,3),dtype='float32')}],
    [{'input0':np.ones((2,3),dtype='float64')}]*2,[{'wrong':np.ones((2,3),dtype='float32')}]*2,
    [{'input0':np.full((2,3),np.nan,dtype='float32')}]*2])
def test_invalid_calibration_rejected(source,tmp_path,calibration):
    with pytest.raises(ex.ExportError):ex.quantize(source[0],tmp_path/'bad.onnx',mode='static',calibration=calibration,dataset_id='test')
    assert not (tmp_path/'bad.onnx').exists()


def test_runtime_validation_and_benchmark(source,tmp_path):
    a,x,m=source;r=ex.Runtime(m);candidate=ex.Runtime(a)
    validation=ex.validate(r,candidate,[x,x*.5])
    assert validation['numerical']['status']=='PASS' and validation['quality']['status']=='NOT_TESTED'
    quality=ex.validate(r,candidate,[(x,np.array([1,1])),(x*.5,np.array([1,1]))],labelled=True)
    assert quality['quality']['status']=='PASS' and quality['quality']['prediction_agreement']==1
    first=ex.benchmark(r,x,warmup=1,iterations=3);second=ex.benchmark(candidate,x,warmup=1,iterations=3)
    assert first['latency']['p99_ms']>0 and first['gpu_vram']['value'] is None
    assert first['host_to_device_ms'] is None
    comparison=ex.compare(first,second,validation=validation)
    assert comparison['comparable'] and comparison['speedup']>0
    second['conditions']['batch_size']=99
    from hyso.analysis._codec import digest
    second['conditions_sha256']=digest(second['conditions'])
    import math
    second['throughput_per_second']=99*3*1000/math.fsum(second['samples_ms'])
    different=ex.compare(first,second)
    assert different['speedup'] is None and not different['comparable']


@pytest.mark.parametrize('language',['tr','en','de','es'])
@pytest.mark.parametrize('format',['json','markdown','html'])
def test_reports_and_advice(language,format):
    row=ex.output_metrics(np.ones(2),np.ones(2));row['user_text']='<script>alert(1)</script>```'
    text=ex.render(row,format=format,language=language)
    assert text and ex.advise(row,language=language)['model_modified'] is False
    if format=='html':assert '<script>' not in text and '&lt;script&gt;' in text
    if format=='json':assert json.loads(text)['language']==language


def test_no_native_runtime_without_admission():
    with pytest.raises(ex.ExportError,match='CUDA_ADMISSION_REQUIRED'):ex.Runtime(Matrix(),device='cuda').run(torch.ones(2,3))


def test_bound_and_named_output_contracts():
    with pytest.raises(ex.ExportError):ex.output_metrics({'a':np.ones(2)},{'b':np.ones(2)})
    with pytest.raises(ex.ExportError):ex.validate(ex.Runtime(Matrix()),ex.Runtime(Matrix()),[],max_batches=2)
    with pytest.raises(ex.ExportError):ex.benchmark(ex.Runtime(Matrix()),torch.ones(2,3),iterations=0)


@pytest.mark.parametrize('backend',['libtorch','torch_export'])
def test_saved_torch_runtime(backend,tmp_path):
    x=torch.ones(2,3);m=Matrix();a=getattr(ex,backend)(m,x,tmp_path/(backend+'.pt'))
    row=ex.validate(ex.Runtime(m),ex.Runtime(a),[x])
    assert row['numerical']['status']=='PASS'


def test_cli_export_registered():
    from hyso.cli.main import build_parser
    parser=build_parser();args=parser.parse_args(['export','dependencies'])
    assert args.action=='dependencies'
    with pytest.raises(SystemExit):parser.parse_args(['export','--help'])


def test_cli_convert_inspect_validate_benchmark(tmp_path,capsys):
    from hyso.cli.main import main
    np.savez(tmp_path/'inputs.npz',input0=np.ones((2,3),dtype='float32'))
    artifact=tmp_path/'cli.onnx';feed=str(tmp_path/'inputs.npz')
    assert main(['export','convert','--factory','test_export_development:Matrix','--example',feed,
                 '--backend','onnx','--execute','--trust-factory','--output',str(artifact)])==0
    manifest=str(artifact)+'.export.json'
    assert main(['export','inspect',manifest])==0
    assert main(['export','benchmark',manifest,'--example',feed,'--execute','--trust-artifact','--iterations','2','--warmup','1'])==0
    assert main(['export','validate',manifest,'--candidate',manifest,'--example',feed,'--execute','--trust-artifact'])==0
    assert 'hyso.export.validation' in capsys.readouterr().out
