"""Small end-to-end public API journeys. Imported only by admitted payloads.

Each journey writes its raw observations BEFORE its acceptance assertions.
No pretrained downloads, intentional overflow, OOM, long run or B4.
"""
from pathlib import Path
import copy
import hashlib
import io
import json
import math
import os
import sys
import traceback

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import hyso
from hyso import performance as perf
from tools.cuda_acceptance.cases import MissingEvidence
from tools.cuda_acceptance.feature_evidence import SafetyViolation,verify_nested_safety

APIS = {
 'U01': 'capture_model_structure; BenchmarkRunner.run',
 'U02': 'torchvision.models(weights=None); capture_model_structure; BenchmarkRunner.run',
 'U03': 'ModelBuilder.from_config/to_json/from_json/build',
 'U04': 'GraphModelBuilder.add/outputs/build',
 'U05': 'Trainer; measure_gradient_diagnostics',
 'U06': 'save_training_checkpoint/load_training_checkpoint; torch.save/load(weights_only=True)',
 'U07': 'evaluate_classification; measure_throughput; evaluate_precision_quality; BenchmarkRunner',
 'U08': 'compare_performance_quality; capture_model_structure; compare_model_structures',
 'U09': 'analysis.plan_analysis/preflight_analysis/execute_analysis; advisor.recommend',
 'U10': 'reporting.create_report; PresentationReport.from_json/to_markdown/to_html',
 'U11': 'hyso CLI; examples/analysis/native_workflow.py',
 'U12': 'BenchmarkRunner; capture_model_structure; advisor.recommend',
 'U13': 'torch.onnx.export; performance.profile_execution_providers',
 'U14': 'ModelBuilder; GraphModelBuilder; BenchmarkRunner; reporting; quality; analysis',
}
_RETAINED_CACHE = {}


class ExternalCNN(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(width, 2)
        self.register_buffer('offset', torch.zeros(1))

    def forward(self, x):
        return self.head(self.features(x).flatten(1)) + self.offset


def data():
    return torch.arange(8*16*16, dtype=torch.float32).reshape(8, 1, 16, 16)/2048, torch.tensor([0,1]*4)


def loader():
    return DataLoader(TensorDataset(*data()), batch_size=4, shuffle=False, num_workers=0)


def tensor_hash(t):
    value = t.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def witness(m):
    return dict(state={k: tensor_hash(v) for k,v in m.state_dict().items()},
        modes={k: v.training for k,v in m.named_modules()},
        parameters={k: dict(identity=id(p), device=str(p.device), requires_grad=p.requires_grad,
            gradient=None if p.grad is None else dict(identity=id(p.grad), hash=tensor_hash(p.grad)))
            for k,p in m.named_parameters()},
        buffers={k: dict(identity=id(p), device=str(p.device)) for k,p in m.named_buffers()})


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Native results already encode nonfinite observations as explicit null+status.
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


class Journey:
    def __init__(self, folder, device, native=None):
        self.folder = Path(folder); self.device = device; self.native = native
        attempt=self.folder.parent if self.folder.name in APIS else self.folder
        self.cache=_RETAINED_CACHE.setdefault((str(attempt.resolve()),device),{})

    def raw(self, name, value):
        write(self.folder / (name + '-raw.json'), value)
        verify_nested_safety(value)
        return value

    def shared(self, name, factory):
        if name not in self.cache: self.cache[name] = factory()
        return self.cache[name]


def benchmark(m, x, device):
    return hyso.BenchmarkRunner(m, device=device, example_input=x,
        warmup_iterations=1, measure_iterations=5, repeats=1,
        measure_cuda_memory=device.startswith('cuda')).run().to_dict()


def measured_forward(j, m, x, name):
    before = witness(m)
    with torch.no_grad(): out = m(x)
    structure = hyso.capture_model_structure(m).to_dict()
    result = benchmark(m, x, j.device)
    after = witness(m)
    d = j.raw(name, dict(shape=list(out.shape), finite=bool(torch.isfinite(out).all()),
        output_device=str(out.device), parameters=sum(p.numel() for p in m.parameters()),
        structure=structure, benchmark=result, before=before, after=after))
    assert d['finite'] and out.device == torch.device(j.device) and len(out) == len(x)
    assert d['parameters'] > 0 and before == after
    times = result['timing']['latency_samples_ms']
    assert len(times) == 1 and len(times[0]) == 5 and all(math.isfinite(t) and t > 0 for t in times[0])
    assert result['configuration']['precision'] == 'fp32'
    if j.device.startswith('cuda'):
        assert result['cuda_memory']['status'] == 'measured' and result['cuda_memory']['peak_allocated_bytes'] > 0
    return d


def U01(j):
    return measured_forward(j, ExternalCNN().to(j.device), data()[0][:2].to(j.device), 'external')


def U02(j):
    try: from torchvision import models
    except ImportError as exc: raise MissingEvidence('torchvision unavailable: ' + str(exc))
    rows = []; blocked = []
    for name in ('resnet18', 'vgg11', 'mobilenet_v3_small', 'efficientnet_b0'):
        constructor = getattr(models, name)
        # Estimate registered storage WITHOUT allocating real model tensors.
        with torch.device('meta'): meta = constructor(weights=None)
        logical = sum(t.numel()*t.element_size() for t in list(meta.parameters())+list(meta.buffers()))
        del meta
        allowance = 3*1024**3
        if j.native:
            allowance = min(allowance, int(j.native.runtime['total_device_bytes'] * .15 * .70))
        if logical * 4 > allowance:
            blocked.append(dict(model=name, status='BLOCKED', logical_bytes=logical,
                estimated_working_bytes=logical*4, allowance=allowance, reason='pre-allocation budget'))
            j.raw('architectures', dict(rows=rows, blocked=blocked)); continue
        m = constructor(weights=None).eval().to(j.device)
        x = torch.linspace(0,1,3*32*32, device=j.device).reshape(1,3,32,32)
        try: rows.append(dict(model=name, weights=None, result=measured_forward(j,m,x,name)))
        except (AssertionError, ValueError, TypeError) as exc:
            rows.append(dict(model=name, status='FAIL', reason=str(exc)))
        del m, x
        j.raw('architectures', dict(rows=rows, blocked=blocked))
    assert not any(r.get('status') == 'FAIL' for r in rows), rows
    if blocked: raise MissingEvidence('architectures budget-blocked; see architectures-raw.json')
    assert len(rows) == 4
    return dict(rows=rows, blocked=blocked)


def make_builder(kind='conv2d'):
    return hyso.ModelBuilder((1,16,16),2).add(kind, in_channels=1, out_channels=4, kernel_size=3, padding=1)


def U03(j):
    rows = []
    # Representative standard, pointwise and depthwise-separable paths.
    configs = [make_builder().to_config(), make_builder('depthwise_separable_conv2d').to_config(),
        hyso.ModelBuilder((1,16,16),2).pointwise_conv2d(in_channels=1,out_channels=4).to_config()]
    for i, config in enumerate(configs):
        b = hyso.ModelBuilder.from_config(config)
        path = j.folder / ('builder-' + str(i) + '.json'); b.to_json(path)
        restored = hyso.ModelBuilder.from_json(path)
        assert restored.to_config() == config
        m = restored.build().to(j.device)
        r = measured_forward(j,m,data()[0][:2].to(j.device),'builder-'+str(i))
        assert r['shape'] == [2,2]
        rows.append(dict(config=config,result=r))
    return dict(rows=rows)


def graph(multi=False):
    inputs = {'image':[1,16,16]}
    if multi: inputs['hint'] = [1,16,16]
    b = hyso.GraphModelBuilder(inputs)
    b.add('left','conv2d',['image'],in_channels=1,out_channels=1,kernel_size=1,padding=0)
    b.add('right','pointwise_conv2d',['image'],in_channels=1,out_channels=1)
    b.add('branch','add',['left','right']).add('skip','add',['image','branch'])
    source = 'skip'
    if multi: b.add('join','add',['skip','hint']); source='join'
    return b.add('head','classification_head',[source],in_features=1,num_classes=2).outputs(logits='head')


def rejection(fn):
    try: fn()
    except (ValueError, TypeError, KeyError, RuntimeError, NotImplementedError, FileNotFoundError) as exc:
        if isinstance(exc, (SafetyViolation,torch.OutOfMemoryError)): raise
        assert str(exc).strip(), 'Error must explain rejected input'
        return dict(error_type=type(exc).__name__, message=str(exc))
    raise AssertionError('invalid user input was accepted')


def U04(j):
    rows=[]
    for multi in (False,True):
        m=graph(multi).build().to(j.device); x=data()[0][:2].to(j.device)
        before=witness(m); inputs={'image':x}
        if multi:inputs['hint']=x.flip(-1)
        out=m(inputs)['logits']; out.sum().backward()
        d=j.raw('graph-'+str(multi),dict(config=graph(multi).to_config(), shape=list(out.shape),
            output_device=str(out.device),finite=bool(torch.isfinite(out).all()),
            gradients=[bool(p.grad is not None and torch.isfinite(p.grad).all()) for p in m.parameters()]))
        assert d['shape']==[2,2] and d['finite'] and all(d['gradients']) and out.device==torch.device(j.device)
        rows.append(d)
    invalid=rejection(lambda:hyso.GraphModelBuilder({'image':[1,16,16]}).add(
        'bad','add',['missing','image']).outputs(logits='bad').build())
    return dict(rows=rows, invalid_link=invalid)


def U05(j):
    from hyso.models.output import GraphTensorOutput
    episodes=[]
    for name in ('external','builder','graph'):
        for amp in ((False,True) if j.device.startswith('cuda') else (False,)):
            torch.manual_seed(123)
            m=ExternalCNN() if name=='external' else make_builder().build() if name=='builder' else GraphTensorOutput(graph().build(),'logits',task='multiclass')
            m.to(j.device)
            opt=torch.optim.SGD(m.parameters(),lr=.03,momentum=.8)
            trainer=hyso.Trainer(m,optimizer=opt,loss_fn=nn.CrossEntropyLoss(),device=j.device,amp=amp)
            before=witness(m)['state']
            d=perf.measure_gradient_diagnostics(trainer,loader(),epochs=1).to_dict()
            changed=before!=witness(m)['state']; applied=d['summary']['applied']
            row=dict(model=name,amp=amp,applied=applied,weights_changed=changed,gradient_result=d)
            episodes.append(row); j.raw('training-episodes',dict(episodes=episodes))
            assert applied>0 and changed and d['status']=='measured'
            source=d['source_p19']; history=trainer.history
            assert math.isfinite(history['train_loss'][-1]) and 0<=history['train_acc'][-1]<=100
            if amp:
                from tools.cuda_acceptance.p25_evidence import verify_amp
                verify_amp(d)
            else:
                assert all(r['applied'] and not r['skipped'] for r in d['observations'])
    return dict(episodes=episodes)


def state_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.allclose(a,b,rtol=1e-5,atol=1e-7)
    if isinstance(a,dict):return a.keys()==b.keys() and all(state_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
    return a==b


def state_values(value):
    if isinstance(value,torch.Tensor):
        return dict(dtype=str(value.dtype),shape=list(value.shape),values=value.detach().cpu().tolist())
    if isinstance(value,dict):return {str(k):state_values(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [state_values(v) for v in value]
    return value


def U06(j):
    from hyso.checkpoint import save_training_checkpoint,load_training_checkpoint
    def components():
        torch.manual_seed(17);m=ExternalCNN().to(j.device)
        o=torch.optim.SGD(m.parameters(),lr=.03,momentum=.8)
        return m,o,torch.optim.lr_scheduler.StepLR(o,step_size=1,gamma=.8)
    def epoch(m,o,s):
        for x,y in loader():
            o.zero_grad();loss=nn.functional.cross_entropy(m(x.to(j.device)),y.to(j.device));loss.backward();o.step()
        s.step()
    full,fo,fs=components();epoch(full,fo,fs);epoch(full,fo,fs)
    partial,po,ps=components();epoch(partial,po,ps)
    path=j.folder/'resume.pth'; save_training_checkpoint(path,partial,optimizer=po,scheduler=ps,completed_epochs=1)
    resumed,ro,rs=components(); loaded=load_training_checkpoint(path,resumed,optimizer=ro,scheduler=rs,map_location=j.device)
    epoch(resumed,ro,rs)
    model_ok=state_equal(full.state_dict(),resumed.state_dict());opt_ok=state_equal(fo.state_dict(),ro.state_dict());schedule_ok=state_equal(fs.state_dict(),rs.state_dict())
    # Explicit safe tensor-only model persistence roundtrip, no unsafe pickle fallback.
    model_path=j.folder/'model-state.pth';torch.save(resumed.state_dict(),model_path)
    restored,_,_=components();restored.load_state_dict(torch.load(model_path,weights_only=True,map_location=j.device))
    d=j.raw('resume-comparison',dict(model_equal=model_ok,optimizer_equal=opt_ok,scheduler_equal=schedule_ok,
        restored_equal=state_equal(restored.state_dict(),resumed.state_dict()),metadata=loaded,
        uninterrupted=witness(full)['state'],resumed=witness(resumed)['state'],rtol=1e-5,atol=1e-7,
        optimizer_states=state_values([fo.state_dict(),ro.state_dict()]),scheduler_states=[fs.state_dict(),rs.state_dict()]))
    assert all(d[k] for k in ('model_equal','optimizer_equal','scheduler_equal','restored_equal'))
    return d


def cohort():
    x,y=data()
    return hyso.EvaluationCohort(x,y.tolist(),sample_ids=[str(i) for i in range(8)],class_names=['a','b'],
        split='user-holdout',preprocessing_id='local-arange-2048-v1')


def U07(j):
    m=ExternalCNN();before=witness(m)
    ev=hyso.evaluate_classification(m,loader(),class_names=['a','b'],split='user-holdout',device=j.device,capture_evidence=True).to_dict()
    resident=copy.deepcopy(m).to(j.device);x=data()[0][:2].to(j.device)
    bench=benchmark(resident,x,j.device)
    throughput=perf.measure_throughput(resident,x,protocol=perf.ThroughputProtocol(device=j.device,
        warmup_iterations=1,batch_count=3,repeats=1)).to_dict()
    precision=hyso.evaluate_precision_quality(m,cohort(),candidate='amp_fp16' if j.native else 'fp64',device=j.device,batch_size=4).to_dict()
    d=j.raw('evaluation',dict(evaluation=ev,benchmark=bench,throughput=throughput,precision=precision,unchanged=before==witness(m)))
    assert sum(ev['accounting']['batch_sizes'])==8 and len(ev['raw']['predictions'])==8
    assert throughput['actual']['completed_batches']==3 and before==witness(m)
    assert precision['sources']['baseline']['accounting']['batch_sizes']==precision['sources']['candidate']['accounting']['batch_sizes']
    return d


def U08(j):
    a,b=ExternalCNN(3),nn.Sequential(nn.Conv2d(1,6,1),nn.SiLU(),nn.Conv2d(6,4,3,padding=1),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(4,2))
    before=[witness(m) for m in (a,b)]
    hashes_before=[perf.measure_footprint(m).to_dict()['provenance']['model']['weights_sha256'] for m in (a,b)]
    structures=[hyso.capture_model_structure(m) for m in (a,b)]
    comparison=hyso.compare_model_structures(*structures).to_dict()
    d=hyso.compare_performance_quality(a,b,cohort(),device=j.device,batch_size=4,iterations=3,
        memory_metric='cuda_allocated_peak_bytes' if j.native else 'logical_model_bytes',allow_cuda_peak_reset=bool(j.native)).to_dict()
    hashes_after=[perf.measure_footprint(m).to_dict()['provenance']['model']['weights_sha256'] for m in (a,b)]
    result=j.raw('two-models',dict(comparison=d,structure_comparison=comparison,
        source_weights_before=hashes_before,source_weights_after=hashes_after,
        parameters=[sum(p.numel() for p in m.parameters()) for m in (a,b)],
        same_cohort=True,before=before,after=[witness(m) for m in (a,b)]))
    assert before==result['after'] and result['parameters'][0]!=result['parameters'][1]
    assert d['analysis']['memory']['status']=='comparable'
    if j.native:
        from tools.user_acceptance.evidence import native_verify
        # The producer's native state witnesses bind both original models to
        # the measured benchmark hashes; original tensor/mode/gradient state
        # was independently checked above.
        native_verify('P29',result,j.native.runtime)
    return result


def retained(j):
    from hyso import analysis as api
    def capture():
        model=ExternalCNN().to(j.device);x=data()[0][:2].to(j.device)
        features=('P02',) if j.native else ('P02','P11')
        sources=[api.NativeBinding('throughput',('external',),('P02',),api.InferenceSource(model,x,
            options=dict(protocol=perf.ThroughputProtocol(device=j.device,warmup_iterations=1,batch_count=3,repeats=1))),
            device=j.device,policy=api.OperationPolicy(allow_capture=True,allow_cuda=bool(j.native))),
            api.NativeBinding('footprint',('external',),('P11',),api.InferenceSource(model,x),
                device=j.device,policy=api.OperationPolicy(allow_capture=True,allow_cuda=bool(j.native)))]
        if j.native:sources=sources[:1]
        plan=api.plan_analysis(api.ModelContext('external'),features=features,execution_mode='native',device=j.device)
        result=api.execute_analysis(api.preflight_analysis(plan,sources),authorized_features=features,explicit_authorization=True)
        j.raw('retained-execution',result.to_dict())
        assert result.to_dict()['status_counts'].get('COMPLETED',0)==len(features), result.to_dict()['status_counts']
        return result.to_unified_report(context=api.AnalysisContext(workload_id='user-external',task='classification',model={},dataset={}))
    return j.shared('retained',capture)


def U09(j):
    from hyso.advisor import recommend
    source=retained(j);before=source.to_dict();advice=recommend(source)
    d=j.raw('advisor',dict(source=before,recommendations=advice.to_dict()))
    assert source.to_dict()==before
    ids={r['record_id'] for r in before['index']['records']}
    assert ids and all(r['validation']['status']=='native_validated' for r in before['index']['records'])
    items=d['recommendations']['recommendations']
    assert items and any(item['retained_record_ids'] for item in items)
    assert all(set(item['retained_record_ids'])<=ids and item['manual_only'] is True and
        item['estimated_improvement'] is None and item['verified_effect']=='not_measured' for item in items)
    # Native recommendation validation binds metrics/references to retained sources.
    from hyso.advisor import RecommendationReport
    assert RecommendationReport(advice.to_dict()).to_dict()==advice.to_dict()
    return d


def U10(j):
    from hyso.reporting import create_report,PresentationReport
    from html.parser import HTMLParser
    source=retained(j);rows=[];reference=None;metric_reference=None
    for language in ('tr','en','de','es'):
        report=create_report(source,language=language)
        path=j.folder/('report-'+language+'.json');report.to_json(path)
        loaded=PresentationReport.from_json(path)
        assert loaded.to_dict()==report.to_dict()
        md=loaded.to_markdown(j.folder/('report-'+language+'.md'))
        html=loaded.to_html(j.folder/('report-'+language+'.html'));HTMLParser().feed(html)
        raw=[s['raw'] for s in report.to_dict()['sources']]
        metrics=[{k:m.get(k) for k in ('value','unit','domain','record_id','feature')}
            for s in report.to_dict()['sources'] for m in s['metrics']]
        if reference is None:reference=raw
        if metric_reference is None:metric_reference=metrics
        assert raw==reference and metrics==metric_reference and len(md)>0 and '<html' in html.lower()
        rows.append(dict(language=language,sources=raw,markdown_bytes=len(md.encode()),html_bytes=len(html.encode())))
    return dict(reports=rows)


def U11(j):
    import subprocess
    repo=Path(__file__).resolve().parents[2];rows=[]
    request=json.loads((j.folder.parent/'request.json').read_text(encoding='utf-8'))
    child_env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(repo),*request['runtime_paths']]),PYTHONNOUSERSITE='1')
    commands=[(['-m','hyso','--help'],0),(['-m','hyso','models','list'],0),
        (['-m','hyso','analysis','plan','--features','P02','P11','--json'],0),
        (['-m','hyso','__invalid_command__'],2),
        ([str(repo/'examples/analysis/native_workflow.py'),'--help'],0),
        ([str(repo/'examples/analysis/native_workflow.py'),'cpu','--features','P11','--execute','--output',str(j.folder/'cli-output')],0)]
    for argv,expected in commands:
        # Child inherits no-breakaway Job and CUDA-invisible CPU environment.
        p=subprocess.run([sys.executable,'-S','-B',*argv],cwd=repo,env=child_env,text=True,capture_output=True,timeout=60)
        rows.append(dict(argv=argv,expected_exit=expected,exit=p.returncode,stdout=p.stdout,stderr=p.stderr))
        j.raw('cli',dict(commands=rows))
        assert p.returncode==expected and (p.stdout.strip() or p.stderr.strip()),rows[-1]
    assert all((j.folder/'cli-output'/n).is_file() for n in ('execution.json','evidence.json','report.json','report.md','report.html'))
    return dict(commands=rows)


def U12(j):
    m=ExternalCNN().to(j.device);m.train();m.features[0].eval()
    for p in m.parameters():p.grad=torch.ones_like(p)
    before=witness(m)
    source_path=Path(__file__).resolve();source_hash=hashlib.sha256(source_path.read_bytes()).hexdigest()
    measurement=measured_forward(j,m,data()[0][:2].to(j.device),'read-only')
    advice=U09(j)
    after=witness(m)
    d=j.raw('integrity',dict(before=before,after=after,source_path=str(source_path),source_sha256=source_hash,
        measurement=measurement,advisor=advice))
    assert before==after and hashlib.sha256(source_path.read_bytes()).hexdigest()==source_hash
    return d


def onnx_model(identity_path=None):
    # PyTorch export is the public exporter used by the existing deployment API.
    if identity_path:
        from tools.cuda_acceptance.native_identity import register_ort_identity
        register_ort_identity(identity_path)
    m=ExternalCNN().eval();x=data()[0][:1];buf=io.BytesIO()
    torch.onnx.export(m,x,buf,input_names=['image'],output_names=['logits'],opset_version=17)
    return buf.getvalue(),{'image':x.numpy()}


def U13(j):
    identity=j.folder/'ort-loaded-runtime.json'
    options=dict(identity_path=str(identity))
    recipe=perf.WorkloadRecipe('tools.user_acceptance.scenarios:onnx_model',options,
        **({'gpu_uuid':j.native.uuid} if j.native else {}))
    provider='CUDAExecutionProvider' if j.native else 'CPUExecutionProvider'
    kwargs=dict(providers=[provider,'CPUExecutionProvider'] if j.native else [provider],expected={'*':provider},optimization='disabled')
    if j.native:
        kwargs.update(runtime_search_path=j.native.plan['runtime']['ort_site'],
            provider_options={provider:{'device_id':0,'gpu_mem_limit':256*1024**2}})
    else:
        from tools.user_acceptance.cpu_onnx import runtime_options
        request=json.loads((j.folder.parent/'request.json').read_text(encoding='utf-8'))
        cpu_options,cpu_manifest=runtime_options(request)
        kwargs.update(cpu_options)
    d=perf.profile_execution_providers(recipe,j.folder/'providers',protocol=perf.ProviderProtocol(
        warmup=1,runs=3,process_memory_bytes=3*1024**3,job_memory_bytes=6*1024**3,max_processes=4),**kwargs).to_dict()
    j.raw('onnx-provider',d)
    a=d['analysis'];assert a['status']=='expected_execution_observed' and a['events'] and all(e['provider']==provider for e in a['events'])
    assert not a['fallback_detected'] and not a['uncovered_source_nodes']
    if j.native:
        from tools.cuda_acceptance.native_identity import validate_ort_identity
        loaded=json.loads(identity.read_text(encoding='utf-8'));planned=j.native.plan['feature_tools']['ort']
        validate_ort_identity(loaded,d,planned)
        return dict(result=d,loaded_runtime=loaded,planned_runtime=planned)
    from tools.user_acceptance.cpu_onnx import verify_cpu_execution
    loaded=json.loads(identity.read_text(encoding='utf-8'))
    j.raw('onnx-runtime-identity',dict(loaded_runtime=loaded,planned_runtime=cpu_manifest))
    verify_cpu_execution(d,loaded,cpu_manifest)
    return dict(result=d,loaded_runtime=loaded,planned_runtime=cpu_manifest)


def U14(j):
    from hyso import analysis as api
    from hyso.reporting import PresentationReport,create_report
    m=ExternalCNN();x=data()[0][:2]
    errors={}
    checks={
      'shape':lambda:hyso.BenchmarkRunner(m,input_shape=(0,1,16,16)),
      'device':lambda:hyso.BenchmarkRunner(m,example_input=x,device='invalid:0'),
      'model_config':lambda:hyso.ModelBuilder.from_config({'invalid':True}),
      'graph_config':lambda:hyso.GraphModelBuilder({'image':[1,16,16]}).add('bad','unknown',['image']).outputs(logits='bad').build(),
      'precision':lambda:hyso.BenchmarkRunner(m,example_input=x,precision='imaginary'),
      'runtime':lambda:perf.ProviderProtocol(runs=-1),
      'runtime_directory':lambda:perf.profile_execution_providers(perf.WorkloadRecipe(
          'tools.user_acceptance.scenarios:onnx_model'),j.folder/'bad-runtime',
          runtime_search_path=str(j.folder/'nonexistent-runtime')),
      'file':lambda:hyso.ModelBuilder.from_json(j.folder/'missing.json'),
      'report':lambda:PresentationReport.from_json(j.folder/'bad-report.json'),
      'comparison':lambda:api.compare_models(None,None,features=['P02']),
      'language':lambda:create_report(retained(j),language='xx'),
    }
    (j.folder/'bad-report.json').write_text('{"format":"invalid"}',encoding='utf-8')
    for name,fn in checks.items():
        errors[name]=rejection(fn);j.raw('invalid-inputs',errors)
    # Actual measured evidence with an incompatible cohort must never produce
    # an authoritative comparison or silently refresh missing measurements.
    evidence=hyso.compare_performance_quality(ExternalCNN(3),ExternalCNN(4),cohort(),iterations=2,batch_size=4).to_dict()
    bad=copy.deepcopy(evidence);bad['sources']['cohort']['sample_ids'][0]='incompatible-cohort-id'
    errors['incompatible_evidence']=rejection(lambda:hyso.QualityExperimentResult(bad))
    j.raw('invalid-measurement',dict(original=evidence,rejected=bad,error=errors['incompatible_evidence']))
    assert len(errors)==len(checks)+1
    return errors


def execute(j, name):
    folder=j.folder/name;folder.mkdir(parents=True,exist_ok=False)
    child=Journey(folder,j.device,j.native)
    row=dict(scenario=name,public_api=APIS[name],model='local untrained CNN / explicit user fixtures',
        device=j.device,expected='public operation, functional measurements and invariants',
        actual=None,status='RUNNING',raw_evidence_path=str(folder),product_defect=None,support_limitation=None)
    write(folder/'result.json',row)
    try:
        observations=globals()[name](child)
        row.update(status='PASS',actual='functional assertions satisfied',checks={'functional_evidence':True},observations=observations)
    except MissingEvidence as exc:
        row.update(status='BLOCKED',actual=str(exc),support_limitation=str(exc));raise
    except (SafetyViolation,torch.OutOfMemoryError) as exc:
        row.update(status='FAIL',actual=traceback.format_exc(),product_defect='safety stop; requires investigation')
        write(j.folder/'user-safety-stop.json',dict(scenario=name,reason=str(exc)))
        if j.device=='cpu':
            import pytest
            pytest.exit('USER_OWNED_SAFETY_STOP: '+str(exc),returncode=80)
        raise
    except Exception as exc:
        row.update(status='FAIL',actual=traceback.format_exc(),product_defect='untriaged: public scenario failure');raise
    finally:write(folder/'result.json',row)
    return row


def gpu_group(ctx):
    torch.set_num_threads(1)
    j=Journey(ctx.folder,'cuda:0',ctx)
    for name in ctx.plan['groups'][ctx_group(ctx)]:
        ctx.run(name,lambda name=name:execute(j,name),scope='bounded_public_user_journey')


def ctx_group(ctx):
    request=json.loads((ctx.folder/'request.json').read_text(encoding='utf-8'))
    return request['group']
