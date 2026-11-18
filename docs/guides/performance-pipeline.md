# Pipeline, stage timing ve fresh-process cold start

P03/P04 instrumented serial pipeline'dır; mevcut BenchmarkRunner resident latency
ve P02 sustained throughput farklı scope'lardır. Kamera/display FPS, overlap
scheduler veya conversion engine bu API'nin kapsamında değildir.

## CPU source/read → preprocess → inference → postprocess

Python bloklarını sırayla çalıştırın. Source/adapters `PipelineSample(id, Tensor)`
döndürür, ID değiştirmez. Queue bounded blocking'dir; doluyken yeni source okunmaz,
drop policy desteklenmez. ID'ler warmup/repeat dahil çağrı boyunca tekil olmalıdır.

```python
from pathlib import Path
import tempfile
import torch
from hyso.performance import (PipelineSample, PipelineStage, PipelineProtocol,
    measure_pipeline, analyze_stages, ColdStartRecipe, ColdStartProtocol,
    measure_cold_start, measure_footprint)

torch.set_num_threads(1)
output_dir = Path(tempfile.mkdtemp(prefix='hyso-pipeline-'))
counter = iter(range(100))
def read():
    index = next(counter)
    return PipelineSample(str(index), torch.full((1, 1, 5, 7), float(index)))
def preprocess(frame):
    resized = torch.nn.functional.interpolate(frame.payload, size=(4, 6)) / 255.
    return PipelineSample(frame.sample_id, resized)
def postprocess(frame):
    return PipelineSample(frame.sample_id, frame.payload.mean((2, 3), keepdim=True))

model = torch.nn.Conv2d(1, 2, 3, padding=1).eval()
pipeline = measure_pipeline(model, read,
    preprocess=PipelineStage('preprocess', preprocess),
    postprocess=PipelineStage('postprocess', postprocess),
    protocol=PipelineProtocol(samples=4, warmup_samples=1, queue_depth=2))
stages = analyze_stages(pipeline)  # raw stage evidence'dan; model tekrar çalışmaz
assert pipeline.to_dict()['actual']['completed'] == 4
assert stages.to_dict()['measurements']['inference']['count'] == 4
pipeline.to_json(output_dir / 'pipeline.json')
pipeline.to_csv(output_dir / 'pipeline.csv')
stages.to_json(output_dir / 'stages.json')
stages.to_csv(output_dir / 'stages.csv')
```

Source CPU tensor'u clone edilir; adapters/output ordinary dense Tensor olmalıdır.
Caller model önceden seçili device/dtype üzerinde olmalıdır, otomatik taşınmaz.
Warmup count gerçek forward'ları içerir, raw measured rows'a eklenmez. Preflight
adıyla fazladan inference yoktur. E2E aynı ID'nin source/read-start→postprocess
completion host wall süresidir; optional delivery ayrı stage ve timestamp'tir,
E2E'ye dahil değildir. Queue wait diğer read/clone/önceki instrumentation sürelerini
içerebilir. Stage süreleri toplanarak E2E üretilmez. P50/P95 descriptive; P99 için
varsayılan n≥1000 ve (1−q)n≥10, aksi null/insufficient_samples. CI üretilmez.

Failure/incomplete/duplicate-ID/state mutation exception verir; başarılı veya
sahte pending=0 sonucu üretilmez. Başarılı blocking scope'ta dropped=0/pending=0
gerçek protokol invariant'ıdır; gerçek frame drop/camera analizi P30'a aittir.

## CUDA stage clocks

```python
def cuda_pipeline_example():
    if not torch.cuda.is_available():
        raise RuntimeError('Native CUDA required; no CPU fallback')
    gpu_model = torch.nn.Conv2d(1, 2, 3, padding=1).to('cuda:0').eval()
    ids = iter(range(20))
    def source():
        return PipelineSample(str(next(ids)), torch.ones(1, 1, 5, 7))
    def transfer(frame):
        return PipelineSample(frame.sample_id, frame.payload.to('cuda:0', non_blocking=True))
    def post(frame):
        return PipelineSample(frame.sample_id, frame.payload.mean((2, 3), keepdim=True))
    stream = torch.cuda.Stream(device=0)
    torch.cuda.synchronize(0)
    with torch.cuda.stream(stream):
        return measure_pipeline(gpu_model, source,
            transfer=PipelineStage('transfer', transfer, device='cuda'),
            postprocess=PipelineStage('postprocess', post, device='cuda'),
            protocol=PipelineProtocol(device='cuda:0', precision='amp_fp16', samples=3))
```

CPU host launch/completion ms ile GPU event ms ayrı alanlardır. CUDA-declared
stage seçili captured current-stream üzerinde start/end event ve end.synchronize
kullanır; post completion öncesinde terminal event drain vardır. Bu izolasyon
profiler overhead'idir; bare latency veya overlap kapasitesi değildir. Hidden
cross-stream/other-device work ve async CPU callable desteklenmez. CPU stage'in
device_event_ms alanı null/not_applicable'dır. Caller parameter/buffer/gradient,
mode, Python/NumPy/Torch CPU/selected CUDA RNG ve stream/device korunur; arbitrary
Python/external side effects rollback edilmez. Allocator peak reset/GC policy
değiştirilmez. Snapshot/finite/queue/event overhead'i scope'ta açıkça belirtilir.

## Gerçek fresh worker cold start

```python
model_path = output_dir / 'state.pt'
artifact = measure_footprint(model, serialize_to=model_path, verify_roundtrip=True)
info = artifact.to_dict()['raw_evidence']['artifact']
recipe = ColdStartRecipe('torch.nn:Conv2d',
    model_kwargs={'in_channels': 1, 'out_channels': 2, 'kernel_size': 3, 'padding': 1},
    input_shape=(1, 1, 5, 7), artifact_path=model_path, artifact_sha256=info['sha256'])
cold = measure_cold_start(recipe,
    protocol=ColdStartProtocol(trials=1, warmup_iterations=1, warm_samples=2))
assert cold.to_dict()['actual']['measured'] == 1
cold.to_json(output_dir / 'cold.json')
cold.to_csv(output_dir / 'cold.csv')
```

Her trial `-I -B` ile ayrı interpreter'dır. Canlı model/tensor/handle gönderilmez;
trusted importable `module:qualified_name` ve plain kwargs recipe kullanılır.
Custom factory gerekiyorsa explicit `factory_search_path` verilir. Factory CPU
model kurmalı, import/construction sırasında CUDA initialize etmemeli veya hidden
artifact loading yapmamalıdır. Factory kodu sandbox değildir. CUDA cold için
recipe device='cuda:0', precision='fp32' veya 'amp_fp16' seçilir; hazırlık worker'a
aittir, parent model/RNG/device etkilenmez. CPU fp32/fp64 desteklenir.

Worker module import, CPU construction/input creation, verified SHA+weights_only
CPU deserialize/strict load, runtime/context preparation ve model/input movement,
first completion, warmup ve ayrı warm-reference phase'leri kaydedilir. Artifact
yoksa load phase null/disabled'dır. Caller artifact'ı unsafe load fallback ile
açılmaz. P11 size/SHA provenance yeniden kullanılır; load/read/hash I/O bu cold
phase'inin tanımlı maliyetidir, gizli tracking I/O değildir.

Parent startup launch→ready-message **receipt** içerir: pipe/thread scheduling
overhead'i vardır. Worker clocks ayrı domain'de tutulur, cross-process timestamp
çıkarımı yapılmaz. Parent total worker output publication/shutdown'ı da içerir.
Windows venv launcher PID'si gerçek Python worker PID'sinden farklı olabilir;
ikisi ayrı kaydedilir. Timeout'ta yalnız bu launcher'ın sahip olduğu process ağacı
sonlandırılır; process wait/reader join/temp cleanup tamamlanır. Hidden descendant
creation/breakaway factory davranışı destek dışıdır. Timeout/crash/unsupported/
failed trials explicit outcome, unobserved phases null'dır; failed trial başarılı
first-inference average'a girmez. OS page-cache coldluğu unknown_not_flushed'dır.

`trials` istenen bağımsız trial sayısıdır: bir trial failed/unsupported/timeout/
crashed olduğunda kalan trial'lar mevcut sözleşmeye göre yine başlatılır. Public
crash-stop seçeneği yoktur; crash-sensitive tek deneme için `trials=1` kullanılır.
Cleanup hatası ise exception olarak yayılır ve sonraki trial başlatılmaz.

Worker timeout launch öncesinden itibaren izlenir; stdout pipe en fazla 10ms
bekleme aralıklarıyla, EOF/newline beklenmeden bounded chunk'larla okunur. Ready
receipt ölçümü parent polling/scheduling maliyetini içerir; bu değişiklik startup
receipt zamanlamasını etkileyebilir. Worker beklemesinden sonra tree termination
ve launcher reap ortak, ek 10 saniyelik cleanup bütçesi kullanır. Cleanup süresi
parent total'a dahildir; deadline aşımı başarılı termination kaydı üretmez.
Bu bir OS-level outer watchdog değildir: process creation ve filesystem çağrıları
bu Python timeout'uyla kesilemez. Root çıktıktan sonra yaşayan hidden/breakaway
descendant'ların öldürüldüğü garanti edilmez; inherited stdout'un açık kalması
artık parent'ı kilitlemez. POSIX yolunda yalnız owned process kill uygulanır.

## Mevcut Experiment Manager ile ölçüm sonrası kayıt

```python
from hyso.experiments import ExperimentManager
root = output_dir / 'runs'
manager = ExperimentManager(root)
experiment = manager.create_experiment('P03-P04-P05')
manager.close()
ExperimentManager.migrate_store(root)  # explicit offline artifact capability
manager = ExperimentManager(root)
run = manager.create_run(experiment['id'], 'benchmark',
    config={'pipeline_protocol': pipeline.to_dict()['protocol'], 'cold_protocol': cold.to_dict()['protocol']},
    provenance={'pipeline_result_id': pipeline.to_dict()['result_id'],
                'cold_result_id': cold.to_dict()['result_id'], 'model_artifact_sha256': info['sha256']})
run.start()
artifact_ids = []
for name in ('pipeline', 'stages', 'cold'):
    record = run.add_artifact(output_dir / (name + '.json'), kind='performance_result', ownership='copy')
    artifact_ids.append(record['id'])
    assert manager.verify_artifact(record['id'])['status'] == 'verified'
for name, value in [('perf.pipeline.p50', pipeline.to_dict()['measurements']['p50_ms']['value']),
                    ('perf.stage.inference.host_mean', stages.to_dict()['measurements']['inference']['host_mean_ms']),
                    ('perf.cold.first_mean', cold.to_dict()['measurements']['first_inference_mean_ms']['value'])]:
    if value is not None:
        run.log_metric(name=name, value=value, unit='ms', source='performance.pipeline_v1', step=0, step_kind='observation')
run.finish()
run_id = run.id
manager.close()
manager = ExperimentManager(root)
assert manager.get_run(run_id).to_dict()['status'] == 'completed'
assert all(manager.verify_artifact(identity)['status'] == 'verified' for identity in artifact_ids)
manager.close()
```

Raw stage/phase evidence JSON artifacts içindedir, milyonlarca metric event
yapılmaz. Measurement sırasında SQLite write yoktur. Sonuç/run/model-artifact
kimlikleri ayrıdır. Existing schema2 artifact/generic metrics kullanılır; yeni
table/migration/run kind yoktur. B1/B2 sonuç/export sözleşmeleri korunur. Future
ONNX/LibTorch/TensorRT/OpenVINO/TFLite/Core ML/YOLO için backend/artifact provenance
açıktır; bu sürüm PyTorch eager dışında backend veya conversion sunmaz.
